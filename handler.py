import base64
import logging
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

try:
    from .inference import OsTraceDetector
except ImportError:
    from inference import OsTraceDetector

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OsTrace API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector: Optional[OsTraceDetector] = None


@app.on_event("startup")
async def startup():
    global detector
    if OsTraceDetector.is_available():
        try:
            detector = OsTraceDetector()
            logger.info("Детектор успешно загружен")
        except Exception as e:
            logger.error(f"Загрузка детектора был неуспешна: {e}")
    else:
        logger.warning("Не найдено .onnx модели")


def encode_image_to_base64(image: np.ndarray) -> str:
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def load_dicom(data):
    if not HAS_PYDICOM:
        raise ImportError("pydicom не установлен")
    ds = pydicom.dcmread(BytesIO(data) if isinstance(data, bytes) else str(data))
    px = ds.pixel_array.astype(np.float32)
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        try:
            c = float(ds.WindowCenter[0] if isinstance(ds.WindowCenter, list) else ds.WindowCenter)
            w = float(ds.WindowWidth[0] if isinstance(ds.WindowWidth, list) else ds.WindowWidth)
            px = np.clip(px, c - w / 2, c + w / 2)
        except (ValueError, TypeError):
            pass
    mn, mx = px.min(), px.max()
    px = (px - mn) / (mx - mn) if mx > mn else np.zeros_like(px)
    if len(px.shape) == 2:
        px = np.stack([px] * 3, axis=-1)
    return px


def generate_heatmap_overlay(image: np.ndarray, predictions: list, alpha: float = 0.4) -> np.ndarray:
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for pred in predictions:
        cx, cy = pred["x"], pred["y"]
        bw, bh = pred["width"], pred["height"]
        conf = pred["confidence"]
        sx = max(bw * 0.6, 20)
        sy = max(bh * 0.6, 20)

        y_coords, x_coords = np.mgrid[0:h, 0:w]
        gaussian = np.exp(
            -((x_coords - cx) ** 2 / (2 * sx ** 2)
              + (y_coords - cy) ** 2 / (2 * sy ** 2))
        )
        heatmap += gaussian * conf

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    if image.max() <= 1.0:
        vis = (image * 255).astype(np.uint8)
    else:
        vis = image.astype(np.uint8)

    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif vis.shape[2] == 3:
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    return cv2.addWeighted(vis, 1 - alpha, heatmap_color, alpha, 0)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if detector is None:
        return JSONResponse({"error": "Модель не загружена"}, status_code=503)

    try:
        image_bytes = await file.read()
        fname = (file.filename or "").lower()
        is_dicom = fname.endswith(".dcm") or fname.endswith(".dicom")

        if is_dicom:
            if not HAS_PYDICOM:
                return JSONResponse({"error": "pydicom не установлен"}, status_code=400)
            image_array = load_dicom(image_bytes)
        else:
            image_array = np.array(
                Image.open(BytesIO(image_bytes)).convert("RGB"),
                dtype=np.float32,
            ) / 255.0

        result = detector.predict(image_array)
        preds = result["predictions"]
        count = result["count_objects"]

        has_fracture = count > 0
        confidence = preds[0]["confidence"] if preds else 0.0

        overlay = generate_heatmap_overlay(image_array, preds)
        processed_image = encode_image_to_base64(overlay)

        return {
            "has_fracture": has_fracture,
            "confidence": round(confidence, 4),
            "processed_image": processed_image,
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": detector is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
