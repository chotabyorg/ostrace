import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

EXPORTED_MODELS_DIR = Path(__file__).parent / "exported_models_v5"


class OsTraceDetector:
    def __init__(self, model_dir: Optional[str] = None):
        if not HAS_ORT:
            raise ImportError(
                "onnxruntime не установлен. "
                "Установите его с помощью команды: pip install onnxruntime"
            )

        self.model_dir = Path(model_dir) if model_dir else EXPORTED_MODELS_DIR
        self.session = None
        self.model_info = {}
        self.input_name = None
        self.input_shape = None
        self._load_model()

    def _load_model(self):
        onnx_files = list(self.model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(
                f"ONNX модель не найдена в {self.model_dir}. "
            )

        model_path = onnx_files[0]
        info_path = self.model_dir / "model_info.json"

        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                self.model_info = json.load(f)

        providers = ["CPUExecutionProvider"]
        try:
            if ort.get_device() == "GPU":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass

        self.session = ort.InferenceSession(str(model_path), providers=providers)

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape
        self.input_h = self.input_shape[2] if len(self.input_shape) >= 3 else 560
        self.input_w = self.input_shape[3] if len(self.input_shape) >= 4 else 560

        output_names = [o.name for o in self.session.get_outputs()]
        logger.info(
            f"ONNX model loaded: {model_path.name}, "
            f"input={self.input_name} shape={self.input_shape}, "
            f"outputs={output_names}"
        )

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        if len(image.shape) == 3 and image.shape[2] >= 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 2:
            gray = image
        else:
            gray = image

        resized = cv2.resize(gray, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        rgb = np.stack([resized, resized, resized], axis=-1)

        blob = rgb.astype(np.float32) / 255.0
        blob = (blob - self.IMAGENET_MEAN) / self.IMAGENET_STD

        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        return blob

    def postprocess(
        self,
        outputs: list,
        orig_w: int,
        orig_h: int,
        conf_threshold: float = 0.01,
        iou_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        dets = outputs[0]
        labels_logits = outputs[1]

        if dets.ndim == 3:
            dets = dets[0]
        if labels_logits.ndim == 3:
            labels_logits = labels_logits[0]

        probs = 1.0 / (1.0 + np.exp(-labels_logits))

        fracture_scores = np.maximum(probs[:, 1], probs[:, 2])

        mask = fracture_scores >= conf_threshold
        if not mask.any():
            return []

        boxes_norm = dets[mask]
        scores = fracture_scores[mask]

        cx = boxes_norm[:, 0] * orig_w
        cy = boxes_norm[:, 1] * orig_h
        w = boxes_norm[:, 2] * orig_w
        h = boxes_norm[:, 3] * orig_h

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        indices = self._nms(
            np.stack([x1, y1, x2, y2], axis=1), scores, iou_threshold
        )

        predictions = []
        for idx in indices:
            predictions.append({
                "x": float(cx[idx]),
                "y": float(cy[idx]),
                "width": float(w[idx]),
                "height": float(h[idx]),
                "confidence": float(scores[idx]),
                "class": "fracture",
                "class_id": 1,
            })

        predictions.sort(key=lambda p: p["confidence"], reverse=True)
        return predictions

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(int(i))

            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.01,
        iou_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        orig_h, orig_w = image.shape[:2]
        blob = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: blob})
        predictions = self.postprocess(
            outputs, orig_w, orig_h, conf_threshold, iou_threshold
        )
        return {
            "predictions": predictions,
            "count_objects": len(predictions),
        }

    @staticmethod
    def is_available(model_dir: Optional[str] = None) -> bool:
        d = Path(model_dir) if model_dir else EXPORTED_MODELS_DIR
        return HAS_ORT and any(d.glob("*.onnx")) if d.exists() else False
