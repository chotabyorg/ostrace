import base64
import logging
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel


from .inference import FractureDetector, encode_image_to_base64
from .gradcam import GradCAMVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OsTrace API",
    description="Fracture detection API with Grad-CAM visualization. Upload a model to start.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector: Optional[FractureDetector] = None
gradcam_visualizer: Optional[GradCAMVisualizer] = None
current_model_name: Optional[str] = None


class PredictRequest(BaseModel):
    image: str
    include_gradcam: bool = True


class PredictResponse(BaseModel):
    has_fracture: bool
    confidence: float
    processed_image: str
    gradcam_image: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    current_model: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    global detector, current_model_name
    detector = None
    current_model_name = None
    logger.info("OsTrace API started. Upload a model to begin.")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=detector is not None and detector.model is not None,
        current_model=current_model_name,
    )


@app.post("/models/upload")
async def upload_model(file: UploadFile = File(...)):
    global detector, gradcam_visualizer, current_model_name
    
    try:
        model_bytes = await file.read()
        model_filename = file.filename or "uploaded_model.keras"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(model_filename).suffix) as tmp:
            tmp.write(model_bytes)
            temp_path = tmp.name
        
        if detector is None:
            config = Config()
            detector = FractureDetector(config)
        
        detector.load_model_from_file(temp_path)
        current_model_name = model_filename
        
        try:
            gradcam_visualizer = GradCAMVisualizer(detector.model)
            logger.info(f"GradCAMVisualizer инициализирован для загруженной модели")
        except Exception as e:
            logger.warning(f"Не удалось инициализировать GradCAMVisualizer: {e}")
            gradcam_visualizer = None
        
        return {
            "status": "success",
            "current_model": current_model_name,
            "config": detector.config.to_dict(),
        }
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Ошибка загрузки модели: {str(e)}")


def generate_gradcam(image: np.ndarray) -> Optional[np.ndarray]:
    if gradcam_visualizer is None:
        return None
    
    try:
        image_size = detector.config.image_size
        image_resized = cv2.resize(image, (image_size, image_size))
        if image_resized.max() > 1.0:
            image_resized = image_resized / 255.0
        
        overlay = gradcam_visualizer.visualize(image_resized)
        
        original_h, original_w = image.shape[:2]
        overlay = cv2.resize(overlay, (original_w, original_h))
        
        return overlay
    except Exception as e:
        logger.error(f"Failed to generate Grad-CAM: {e}")
        return None


@app.post("/predict", response_model=PredictResponse)
async def predict_fracture(request: PredictRequest):
    if detector is None or detector.model is None:
        raise HTTPException(status_code=503, detail="No model loaded. Please upload a model first.")
    
    try:
        image_data = base64.b64decode(request.image)
        image = np.array(Image.open(BytesIO(image_data)).convert("RGB"), dtype=np.float32) / 255.0
        
        result = detector.predict(image, return_visualization=True)
        
        gradcam_base64 = None
        if request.include_gradcam:
            gradcam_overlay = generate_gradcam(image)
            if gradcam_overlay is not None:
                gradcam_base64 = encode_image_to_base64(gradcam_overlay)
        
        return PredictResponse(
            has_fracture=result["has_fracture"],
            confidence=result["confidence"],
            processed_image=result["processed_image"],
            gradcam_image=gradcam_base64,
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/upload", response_model=PredictResponse)
async def predict_fracture_upload(
    file: UploadFile = File(...),
    include_gradcam: bool = Form(True)
):
    if detector is None or detector.model is None:
        raise HTTPException(status_code=503, detail="No model loaded. Please upload a model first.")
    
    try:
        image_bytes = await file.read()
        image = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"), dtype=np.float32) / 255.0
        
        result = detector.predict(image, return_visualization=True)
        
        gradcam_base64 = None
        if include_gradcam:
            gradcam_overlay = generate_gradcam(image)
            if gradcam_overlay is not None:
                gradcam_base64 = encode_image_to_base64(gradcam_overlay)
        
        return PredictResponse(
            has_fracture=result["has_fracture"],
            confidence=result["confidence"],
            processed_image=result["processed_image"],
            gradcam_image=gradcam_base64,
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/base64", response_model=PredictResponse)
async def predict_fracture_base64(
    image: str = Form(..., description="Base64 encoded image (with or without data URI prefix)"),
    include_gradcam: bool = Form(True)
):
    if detector is None or detector.model is None:
        raise HTTPException(status_code=503, detail="No model loaded. Please upload a model first.")
    
    try:
        if "," in image:
            image = image.split(",")[1]
        
        image_data = base64.b64decode(image)
        image_array = np.array(Image.open(BytesIO(image_data)).convert("RGB"), dtype=np.float32) / 255.0
        
        result = detector.predict(image_array, return_visualization=True)
        
        gradcam_base64 = None
        if include_gradcam:
            gradcam_overlay = generate_gradcam(image_array)
            if gradcam_overlay is not None:
                gradcam_base64 = encode_image_to_base64(gradcam_overlay)
        
        return PredictResponse(
            has_fracture=result["has_fracture"],
            confidence=result["confidence"],
            processed_image=result["processed_image"],
            gradcam_image=gradcam_base64,
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
