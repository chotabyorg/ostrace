import base64
import logging
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from .inference import OsTraceDetector
    from .safety import (
        classify_upload,
        decode_image_bytes,
        get_cors_origins,
        read_limited_upload,
        validate_dicom_dimensions,
    )
except ImportError:
    from inference import OsTraceDetector
    from safety import (
        classify_upload,
        decode_image_bytes,
        get_cors_origins,
        read_limited_upload,
        validate_dicom_dimensions,
    )

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

detector: Optional[OsTraceDetector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    if OsTraceDetector.is_available():
        try:
            detector = OsTraceDetector()
            logger.info("Детектор успешно загружен")
        except Exception as e:
            logger.error(f"Загрузка детектора был неуспешна: {e}")
    else:
        logger.warning("Не найдено .onnx модели")
    yield


app = FastAPI(title="OsTrace API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        image_bytes = await read_limited_upload(file)
        file_kind = classify_upload(file.filename, file.content_type)

        if file_kind == "dicom":
            if not HAS_PYDICOM:
                return JSONResponse({"error": "pydicom не установлен"}, status_code=400)
            validate_dicom_dimensions(image_bytes)
            image_array = load_dicom(image_bytes)
        else:
            image_array = decode_image_bytes(image_bytes)

        result = detector.predict(image_array)
        preds = result.get("predictions", [])
        count = result.get("count_objects", 0)
        original_base64 = encode_image_to_base64(image_array)

        return {
            "count_objects": count,
            "predictions": preds,
            "original_image": original_base64,
        }

    except HTTPException:
        raise
    except Exception:
        logger.error("Prediction error")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse({"error": "Prediction failed. Check the uploaded file and server logs."}, status_code=400)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "detector_ready": detector is not None,
        "model_loaded": detector is not None,
        "dicom_support": HAS_PYDICOM,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
