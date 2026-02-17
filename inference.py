import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

import cv2
import numpy as np
import tensorflow as tf
import keras
from PIL import Image

from .model import build_multitask_model, build_binary_classifier, SpatialAttention, ChannelAttention
from .losses import combined_segmentation_loss, iou_metric, WarmupSchedule

CUSTOM_OBJECTS = {
    "combined_segmentation_loss": combined_segmentation_loss,
    "iou_metric": iou_metric,
    "SpatialAttention": SpatialAttention,
    "ChannelAttention": ChannelAttention,
    "WarmupSchedule": WarmupSchedule,
}


def detect_model_type(model: keras.Model) -> str:
    if isinstance(model.output, dict):
        if "classification" in model.output and "segmentation" in model.output:
            return "multitask"
        elif "classification" in model.output:
            return "binary_dict"
    elif isinstance(model.output, (list, tuple)):
        if len(model.output) == 2:
            return "multitask"
        elif len(model.output) == 1:
            return "binary"

    return "binary"


try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def encode_image_to_base64(image: Union[np.ndarray, bytes]) -> str:
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("utf-8")

    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    byte_data = buffer.tobytes()

    return base64.b64encode(byte_data).decode("utf-8")


def decode_base64_to_image(base64_str: str) -> np.ndarray:
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    byte_data = base64.b64decode(base64_str)

    image = Image.open(BytesIO(byte_data))
    image = image.convert("RGB")

    return np.array(image, dtype=np.float32) / 255.0


def load_dicom(dicom_path: Union[str, Path, bytes]) -> np.ndarray:
    if not HAS_PYDICOM:
        raise ImportError(
            "pydicom необходим для поддержки DICOM. Установите: pip install pydicom"
        )

    if isinstance(dicom_path, bytes):
        from io import BytesIO

        dicom_data = pydicom.dcmread(BytesIO(dicom_path))
    else:
        dicom_data = pydicom.dcmread(str(dicom_path))

    pixel_array = dicom_data.pixel_array.astype(np.float32)

    if hasattr(dicom_data, "WindowCenter") and hasattr(dicom_data, "WindowWidth"):
        try:
            if isinstance(dicom_data.WindowCenter, list):
                center = float(dicom_data.WindowCenter[0])
            else:
                center = float(dicom_data.WindowCenter)

            if isinstance(dicom_data.WindowWidth, list):
                width = float(dicom_data.WindowWidth[0])
            else:
                width = float(dicom_data.WindowWidth)

            min_val = center - width / 2
            max_val = center + width / 2
            pixel_array = np.clip(pixel_array, min_val, max_val)
        except (ValueError, TypeError):
            pass

    pixel_min = pixel_array.min()
    pixel_max = pixel_array.max()
    if pixel_max > pixel_min:
        pixel_array = (pixel_array - pixel_min) / (pixel_max - pixel_min)
    else:
        pixel_array = np.zeros_like(pixel_array)

    if len(pixel_array.shape) == 2:
        pixel_array = np.stack([pixel_array] * 3, axis=-1)

    return pixel_array


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    if mask.dtype != np.uint8:
        mask = (np.clip(mask, 0, 1) * 255).astype(np.uint8)

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 0] = (mask_resized > 127) * color[0]
    colored_mask[:, :, 1] = (mask_resized > 127) * color[1]
    colored_mask[:, :, 2] = (mask_resized > 127) * color[2]

    mask_binary = (mask_resized > 127).astype(np.float32) / 255.0
    overlay = (
        image * (1 - alpha * mask_binary[:, :, np.newaxis])
        + colored_mask * alpha * mask_binary[:, :, np.newaxis]
    ).astype(np.uint8)

    return overlay


class FractureDetector:
    def __init__(
        self,
        config: Optional[Config] = None,
        model_path: Optional[str] = None,
    ):
        self.config = config or Config()

        if model_path:
            self.config.model_path = model_path

        self.model = None
        self._model_name = self.config.model_name
        self._model_type = "binary"
        self._load_model()

    def _load_model(self):
        path = self.config.model_path

        if path and Path(path).exists():
            print(f"[ИНФЕРЕНС] Загрузка модели из: {path}")
            print(f"[ИНФЕРЕНС] Пользовательские объекті: {list(CUSTOM_OBJECTS.keys())}")

            import sys
            import io
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                old_stderr = sys.stderr
                sys.stderr = io.StringIO()

                try:
                    self.model = keras.models.load_model(
                        path,
                        custom_objects=CUSTOM_OBJECTS,
                        compile=False,
                    )
                    sys.stderr = old_stderr
                    self._model_type = detect_model_type(self.model)
                    print(
                        f"[ИНФЕРЕНС] Модель загружена успешно. "
                        f"Тип: {self._model_type}. "
                        f"Входная форма: {self.model.input_shape}"
                    )
                except Exception as e:
                    sys.stderr = old_stderr
                    error_type = type(e).__name__

                    print(f"[ОШИБКА] {error_type}")
                    print(f"[ОШИБКА] Подробности в error.log")

                    raise

    def load_model_from_file(self, model_path: str):
        self.config.model_path = model_path
        self._model_name = Path(model_path).stem
        self._load_model()
        print(f"[ИНФЕРЕНС] Модель загружена из файла: {model_path}")

    @property
    def current_model_name(self) -> Optional[str]:
        return self._model_name

    @property
    def model_type(self) -> str:
        return self._model_type

    def preprocess_image(self, image: np.ndarray) -> tf.Tensor:
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        if image.max() > 1.0:
            image = image / 255.0

        image_resized = tf.image.resize(
            image, [self.config.image_size, self.config.image_size]
        )

        if len(image_resized.shape) == 3:
            image_resized = tf.expand_dims(image_resized, 0)

        return image_resized

    def predict(
        self,
        image: Union[np.ndarray, str, bytes],
        return_visualization: bool = True,
    ) -> Dict:
        if isinstance(image, str):
            if image.startswith("data:image"):
                image = decode_base64_to_image(image)
            else:
                image = (
                    np.array(Image.open(image).convert("RGB"), dtype=np.float32)
                    / 255.0
                )
        elif isinstance(image, bytes):
            image = (
                np.array(
                    Image.open(BytesIO(image)).convert("RGB"), dtype=np.float32
                )
                / 255.0
            )

        original_image = image.copy()

        input_tensor = self.preprocess_image(image)

        predictions = self.model(input_tensor, training=False)

        segmentation_mask = None

        if self._model_type == "multitask":
            if isinstance(predictions, dict):
                classification_score = float(
                    predictions["classification"].numpy()[0, 0]
                )
                segmentation_mask = predictions["segmentation"].numpy()[0, :, :, 0]
            else:
                classification_score = float(predictions[0].numpy()[0, 0])
                segmentation_mask = predictions[1].numpy()[0, :, :, 0]
        elif self._model_type == "binary_dict":
            classification_score = float(
                predictions["classification"].numpy()[0, 0]
            )
        else:
            pred_array = predictions.numpy()
            if pred_array.ndim == 2:
                classification_score = float(pred_array[0, 0])
            else:
                classification_score = float(pred_array[0])

        has_fracture = classification_score >= self.config.threshold

        result = {
            "has_fracture": has_fracture,
            "confidence": classification_score,
        }

        if return_visualization:
            if (
                has_fracture
                and segmentation_mask is not None
                and np.any(segmentation_mask > 0.5)
            ):
                mask_binary = (segmentation_mask > 0.5).astype(np.float32)
                visualization = overlay_mask_on_image(original_image, mask_binary)
            else:
                if original_image.dtype != np.uint8:
                    visualization = (np.clip(original_image, 0, 1) * 255).astype(
                        np.uint8
                    )
                else:
                    visualization = original_image

            result["processed_image"] = encode_image_to_base64(visualization)

        return result

    def predict_batch(
        self,
        images: list,
        return_visualization: bool = True,
    ) -> list:
        return [self.predict(img, return_visualization) for img in images]
