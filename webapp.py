import base64
import logging
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from .model import SpatialAttention, ChannelAttention
from .losses import combined_segmentation_loss, iou_metric, WarmupSchedule
from .inference import FractureDetector, encode_image_to_base64, load_dicom
from .gradcam import GradCAMVisualizer

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

webapp = FastAPI(
    title="OsTrace Web",
    description="Web interface for fracture detection",
    version="1.0.0",
)

webapp.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector: Optional[FractureDetector] = None
gradcam_visualizer: Optional[GradCAMVisualizer] = None
current_model_name: Optional[str] = None


@webapp.on_event("startup")
async def startup_event():
    global detector, current_model_name
    detector = None
    current_model_name = None
    logger.info("OsTrace Web started. Upload a model to begin.")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OsTrace - Fracture Detection</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            padding: 40px 0;
        }
        
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #888;
            font-size: 1.1rem;
        }
        
        .model-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 20px 40px;
            margin: 20px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        .model-section label {
            color: #888;
            font-size: 1rem;
        }
        
        .model-upload-btn {
            background: linear-gradient(90deg, #28a745, #51cf66);
            color: white;
            border: none;
            padding: 10px 25px;
            font-size: 1rem;
            border-radius: 20px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .model-upload-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 20px rgba(40, 167, 69, 0.3);
        }
        
        .current-model {
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid #00d4ff;
            border-radius: 8px;
            padding: 5px 15px;
            font-size: 0.9rem;
            color: #00d4ff;
        }
        
        .backbone-select {
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 8px 15px;
            border-radius: 8px;
            font-size: 0.9rem;
        }
        
        .upload-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 40px;
            margin: 20px 0;
            border: 2px dashed rgba(255, 255, 255, 0.2);
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .upload-section:hover {
            border-color: #00d4ff;
            background: rgba(255, 255, 255, 0.08);
        }
        
        .upload-section.dragover {
            border-color: #00d4ff;
            background: rgba(0, 212, 255, 0.1);
        }
        
        .file-input {
            display: none;
        }
        
        .upload-btn {
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1rem;
            border-radius: 30px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin: 5px;
        }
        
        .upload-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
        }
        
        .preview-container {
            margin: 20px 0;
            display: none;
        }
        
        .preview-container.active {
            display: block;
        }
        
        .preview-image {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            margin: 10px 0;
        }
        
        .analyze-btn {
            background: #28a745;
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1rem;
            border-radius: 30px;
            cursor: pointer;
            margin-top: 20px;
            transition: transform 0.2s, background 0.2s;
        }
        
        .analyze-btn:hover {
            background: #218838;
            transform: scale(1.05);
        }
        
        .analyze-btn:disabled {
            background: #666;
            cursor: not-allowed;
            transform: none;
        }
        
        .options {
            margin: 20px 0;
            display: flex;
            justify-content: center;
            gap: 20px;
        }
        
        .option-label {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
        }
        
        .option-label input {
            width: 18px;
            height: 18px;
        }
        
        .results-section {
            display: none;
            margin: 30px 0;
        }
        
        .results-section.active {
            display: block;
        }
        
        .result-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 30px;
            margin: 20px 0;
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .result-status {
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .result-status.fracture {
            color: #ff6b6b;
        }
        
        .result-status.normal {
            color: #51cf66;
        }
        
        .confidence-bar {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        .confidence-fill.high {
            background: linear-gradient(90deg, #ff6b6b, #ff8787);
        }
        
        .confidence-fill.low {
            background: linear-gradient(90deg, #51cf66, #69db7c);
        }
        
        .images-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .image-card {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 15px;
            padding: 15px;
        }
        
        .image-card h3 {
            margin-bottom: 10px;
            color: #00d4ff;
        }
        
        .image-card img {
            width: 100%;
            border-radius: 10px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top-color: #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .error-message {
            background: rgba(255, 107, 107, 0.2);
            border: 1px solid #ff6b6b;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            display: none;
        }
        
        .error-message.active {
            display: block;
        }
        
        .format-info {
            color: #888;
            font-size: 0.9rem;
            margin-top: 15px;
        }
        
        footer {
            text-align: center;
            padding: 40px 0;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>OsTrace</h1>
            <p class="subtitle">AI-Powered Fracture Detection & Localization</p>
        </header>
        
        <div class="model-section">
            <label>Загрузить модель:</label>
            <input type="file" id="modelFileInput" class="file-input" accept=".keras,.h5">
            <button class="model-upload-btn" onclick="document.getElementById('modelFileInput').click()">
                Выбрать файл модели
            </button>
            <span class="current-model" id="currentModelLabel">Модель не загружена</span>
        </div>
        
        <div class="upload-section" id="dropZone">
            <input type="file" id="fileInput" class="file-input" accept="image/*,.dcm,.dicom">
            <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                Choose X-Ray Image
            </button>
            <p class="format-info">Supported formats: JPEG, PNG, DICOM (.dcm)</p>
            
            <div class="preview-container" id="previewContainer">
                <img id="previewImage" class="preview-image" alt="Preview">
                <div class="options">
                    <label class="option-label">
                        <input type="checkbox" id="includeGradcam" checked>
                        Show Grad-CAM visualization
                    </label>
                </div>
                <button class="analyze-btn" id="analyzeBtn" onclick="analyzeImage()">
                    Analyze Image
                </button>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing image...</p>
        </div>
        
        <div class="error-message" id="errorMessage"></div>
        
        <div class="results-section" id="resultsSection">
            <div class="result-card">
                <div class="result-header">
                    <span class="result-status" id="resultStatus"></span>
                    <span id="confidenceText"></span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidenceFill"></div>
                </div>
            </div>
            
            <div class="images-grid" id="imagesGrid">
                <div class="image-card" id="processedCard">
                    <h3>Segmentation Overlay</h3>
                    <img id="processedImage" alt="Processed">
                </div>
                <div class="image-card" id="gradcamCard">
                    <h3>Grad-CAM Attention</h3>
                    <img id="gradcamImage" alt="Grad-CAM">
                </div>
            </div>
        </div>
        
        <footer>
            <p>OsTrace v1.0.0 - Multi-task Fracture Detection Model</p>
        </footer>
    </div>
    
    <script>
        let selectedFile = null;
        let isDicom = false;
        let currentModel = null;
        
        document.getElementById('modelFileInput').addEventListener('change', function(e) {
            uploadModel(e.target.files[0]);
        });
        
        async function uploadModel(file) {
            if (!file) return;
            
            const label = document.getElementById('currentModelLabel');
            label.textContent = 'Загрузка...';
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/api/models/upload', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Ошибка загрузки модели');
                }
                
                const data = await response.json();
                currentModel = data.current_model;
                label.textContent = 'Модель: ' + currentModel;
                console.log('Модель загружена:', data);
            } catch (error) {
                console.error('Ошибка загрузки модели:', error);
                label.textContent = 'Ошибка: ' + error.message;
                setTimeout(() => {
                    label.textContent = currentModel ? 'Модель: ' + currentModel : 'Модель не загружена';
                }, 2000);
            }
        }
        
        document.getElementById('fileInput').addEventListener('change', function(e) {
            handleFile(e.target.files[0]);
        });
        
        const dropZone = document.getElementById('dropZone');
        
        dropZone.addEventListener('dragover', function(e) {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', function(e) {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', function(e) {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            handleFile(e.dataTransfer.files[0]);
        });
        
        function handleFile(file) {
            if (!file) {
                showError('Please select a file');
                return;
            }
            
            const fileName = file.name.toLowerCase();
            isDicom = fileName.endsWith('.dcm') || fileName.endsWith('.dicom');
            
            if (!isDicom && !file.type.startsWith('image/')) {
                showError('Please select a valid image file or DICOM file');
                return;
            }
            
            selectedFile = file;
            
            if (isDicom) {
                document.getElementById('previewImage').src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMjIyIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZpbGw9IiM4ODgiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+RElDT00gRmlsZTwvdGV4dD48L3N2Zz4=';
                document.getElementById('previewContainer').classList.add('active');
                document.getElementById('resultsSection').classList.remove('active');
                document.getElementById('errorMessage').classList.remove('active');
            } else {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('previewImage').src = e.target.result;
                    document.getElementById('previewContainer').classList.add('active');
                    document.getElementById('resultsSection').classList.remove('active');
                    document.getElementById('errorMessage').classList.remove('active');
                };
                reader.readAsDataURL(file);
            }
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('errorMessage');
            errorDiv.textContent = message;
            errorDiv.classList.add('active');
        }
        
        async function analyzeImage() {
            if (!selectedFile) {
                showError('Please select an image first');
                return;
            }
            
            if (!currentModel) {
                showError('Please upload a model first');
                return;
            }
            
            const analyzeBtn = document.getElementById('analyzeBtn');
            const loading = document.getElementById('loading');
            const resultsSection = document.getElementById('resultsSection');
            const errorMessage = document.getElementById('errorMessage');
            
            analyzeBtn.disabled = true;
            loading.classList.add('active');
            resultsSection.classList.remove('active');
            errorMessage.classList.remove('active');
            
            try {
                const base64 = await new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onload = () => resolve(reader.result.split(',')[1]);
                    reader.onerror = reject;
                    reader.readAsDataURL(selectedFile);
                });
                
                const includeGradcam = document.getElementById('includeGradcam').checked;
                
                const formData = new FormData();
                formData.append('image', base64);
                formData.append('include_gradcam', includeGradcam);
                formData.append('is_dicom', isDicom);
                
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || `Server error: ${response.statusText}`);
                }
                
                const result = await response.json();
                displayResults(result, includeGradcam);
                
            } catch (error) {
                showError(`Analysis failed: ${error.message}`);
            } finally {
                analyzeBtn.disabled = false;
                loading.classList.remove('active');
            }
        }
        
        function displayResults(result, includeGradcam) {
            const resultsSection = document.getElementById('resultsSection');
            const resultStatus = document.getElementById('resultStatus');
            const confidenceText = document.getElementById('confidenceText');
            const confidenceFill = document.getElementById('confidenceFill');
            const processedImage = document.getElementById('processedImage');
            const gradcamImage = document.getElementById('gradcamImage');
            const gradcamCard = document.getElementById('gradcamCard');
            
            if (result.has_fracture) {
                resultStatus.textContent = 'Fracture Detected';
                resultStatus.className = 'result-status fracture';
            } else {
                resultStatus.textContent = 'No Fracture Detected';
                resultStatus.className = 'result-status normal';
            }
            
            const confidencePercent = (result.confidence * 100).toFixed(1);
            confidenceText.textContent = `Confidence: ${confidencePercent}%`;
            
            confidenceFill.style.width = `${confidencePercent}%`;
            if (result.has_fracture) {
                confidenceFill.className = 'confidence-fill high';
            } else {
                confidenceFill.className = 'confidence-fill low';
            }
            
            processedImage.src = `data:image/jpeg;base64,${result.processed_image}`;
            
            if (includeGradcam && result.gradcam_image) {
                gradcamImage.src = `data:image/jpeg;base64,${result.gradcam_image}`;
                gradcamCard.style.display = 'block';
            } else {
                gradcamCard.style.display = 'none';
            }
            
            resultsSection.classList.add('active');
        }
    </script>
</body>
</html>
"""


@webapp.get("/", response_class=HTMLResponse)
async def root():
    return HTML_TEMPLATE


@webapp.post("/api/predict")
async def api_predict(
    image: str = Form(...),
    include_gradcam: bool = Form(True),
    is_dicom: bool = Form(False)
):
    if detector is None or detector.model is None:
        return JSONResponse({"error": "No model loaded. Please upload a model first."}, status_code=503)
    
    try:
        image_data = base64.b64decode(image)
        
        if is_dicom:
            if not HAS_PYDICOM:
                return JSONResponse(
                    {"error": "DICOM support not available. Install pydicom: pip install pydicom"},
                    status_code=400
                )
            image_array = load_dicom(image_data)
        else:
            image_array = np.array(
                Image.open(BytesIO(image_data)).convert("RGB"),
                dtype=np.float32
            ) / 255.0
        
        result = detector.predict(image_array, return_visualization=True)
        
        gradcam_base64 = None
        if include_gradcam and gradcam_visualizer is not None:
            try:
                image_size = detector.config.image_size
                image_resized = cv2.resize(image_array, (image_size, image_size))
                if image_resized.max() > 1.0:
                    image_resized = image_resized / 255.0
                
                if image_resized is not None and image_resized.size > 0:
                    overlay = gradcam_visualizer.visualize(image_resized)
                    original_h, original_w = image_array.shape[:2]
                    overlay = cv2.resize(overlay, (original_w, original_h))
                    gradcam_base64 = encode_image_to_base64(overlay)
            except Exception as e:
                logger.error(f"Grad-CAM generation failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        return {
            "has_fracture": result["has_fracture"],
            "confidence": result["confidence"],
            "processed_image": result["processed_image"],
            "gradcam_image": gradcam_base64,
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=400)


@webapp.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    include_gradcam: bool = Form(True)
):
    if detector is None or detector.model is None:
        return JSONResponse({"error": "No model loaded. Please upload a model first."}, status_code=503)
    
    try:
        image_bytes = await file.read()
        filename = file.filename.lower() if file.filename else ""
        is_dicom = filename.endswith('.dcm') or filename.endswith('.dicom')
        
        if is_dicom:
            if not HAS_PYDICOM:
                return JSONResponse(
                    {"error": "DICOM support not available. Install pydicom: pip install pydicom"},
                    status_code=400
                )
            image_array = load_dicom(image_bytes)
        else:
            image_array = np.array(
                Image.open(BytesIO(image_bytes)).convert("RGB"),
                dtype=np.float32
            ) / 255.0
        
        result = detector.predict(image_array, return_visualization=True)
        
        gradcam_base64 = None
        if include_gradcam and gradcam_visualizer is not None:
            try:
                image_size = detector.config.image_size
                image_resized = cv2.resize(image_array, (image_size, image_size))
                overlay = gradcam_visualizer.visualize(image_resized)
                original_h, original_w = image_array.shape[:2]
                overlay = cv2.resize(overlay, (original_w, original_h))
                gradcam_base64 = encode_image_to_base64(overlay)
            except Exception as e:
                logger.error(f"Grad-CAM generation failed: {e}")
        
        return {
            "has_fracture": result["has_fracture"],
            "confidence": result["confidence"],
            "processed_image": result["processed_image"],
            "gradcam_image": gradcam_base64,
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=400)


@webapp.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": detector is not None and detector.model is not None,
        "gradcam_available": gradcam_visualizer is not None,
        "dicom_support": HAS_PYDICOM,
        "current_model": current_model_name,
    }


@webapp.post("/api/models/upload")
async def api_upload_model(file: UploadFile = File(...)):
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
        error_type = type(e).__name__
        error_msg = str(e)
        
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "... [обрезано]"
        
        logger.error(f"Ошибка загрузки модели: {error_type}: {error_msg}")
        
        with open("error.log", "w", encoding="utf-8") as f:
            import traceback
            f.write(f"Тип ошибки: {error_type}\n")
            f.write(f"Сообщение (первые 2000 символов): {str(e)[:2000]}\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
        
        return JSONResponse({"detail": f"{error_type}: {error_msg}"}, status_code=400)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(webapp, host="0.0.0.0", port=8080)