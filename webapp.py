import base64
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from .inference import OsTraceDetector

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

webapp = FastAPI(
    title="OsTrace Web",
    description="On-prem fracture detection",
    version="2.0.0",
)

webapp.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

offline_detector: Optional[OsTraceDetector] = None


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
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def load_dicom(dicom_path):
    if not HAS_PYDICOM:
        raise ImportError("pydicom required for DICOM support. Install: pip install pydicom")
    import pydicom
    if isinstance(dicom_path, bytes):
        dicom_data = pydicom.dcmread(BytesIO(dicom_path))
    else:
        dicom_data = pydicom.dcmread(str(dicom_path))
    pixel_array = dicom_data.pixel_array.astype(np.float32)
    if hasattr(dicom_data, "WindowCenter") and hasattr(dicom_data, "WindowWidth"):
        try:
            center = float(dicom_data.WindowCenter[0] if isinstance(dicom_data.WindowCenter, list) else dicom_data.WindowCenter)
            width = float(dicom_data.WindowWidth[0] if isinstance(dicom_data.WindowWidth, list) else dicom_data.WindowWidth)
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


@webapp.on_event("startup")
async def startup_event():
    global offline_detector
    if OsTraceDetector.is_available():
        try:
            offline_detector = OsTraceDetector()
            logger.info("ONNX detector loaded successfully.")
        except Exception as e:
            logger.error(f"ONNX detector init failed: {e}")
            offline_detector = None
    else:
        logger.warning("No ONNX model found. Place weights.onnx in exported_models_v5/")
    logger.info("OsTrace Web started.")


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
        
        .status-bar {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 15px 30px;
            margin: 20px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }

        .status-dot.ok { background: #51cf66; }
        .status-dot.error { background: #ff6b6b; }
        
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
            background: linear-gradient(90deg, #7b2cbf, #9d4edd);
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
            transform: scale(1.05);
            box-shadow: 0 10px 30px rgba(123, 44, 191, 0.3);
        }
        
        .analyze-btn:disabled {
            background: #666;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
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
            border-left: 4px solid #7b2cbf;
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
            color: #9d4edd;
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
            color: #9d4edd;
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
            <p class="subtitle">AI-Powered Fracture Detection &amp; Localization</p>
        </header>
        
        <div class="status-bar">
            <span class="status-dot" id="statusDot"></span>
            <span id="statusText" style="color: #888; font-size: 0.95rem;">Проверка...</span>
        </div>
        
        <div class="upload-section" id="dropZone">
            <input type="file" id="fileInput" class="file-input" accept="image/*,.dcm,.dicom">
            <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                Choose X-Ray Image
            </button>
            <p class="format-info">Supported formats: JPEG, PNG, DICOM (.dcm)</p>
            
            <div class="preview-container" id="previewContainer">
                <img id="previewImage" class="preview-image" alt="Preview">
                <br>
                <button class="analyze-btn" id="analyzeBtn" onclick="analyzeImage()">
                    🔬 Analyze
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
                    <span class="result-status">Detection Results</span>
                    <span id="countBadge" style="background: linear-gradient(90deg, #7b2cbf, #9d4edd); padding: 5px 15px; border-radius: 20px; font-size: 0.9rem;"></span>
                </div>
                <div style="margin: 15px 0;">
                    <label style="color: #aaa; font-size: 0.9rem;">Confidence Threshold: <span id="thresholdValue">0.05</span></label>
                    <input type="range" id="confidenceThreshold" min="0" max="100" value="5" style="width: 100%; accent-color: #9d4edd; margin-top: 5px;">
                </div>
                <div style="margin: 15px 0;">
                    <label style="color: #aaa; font-size: 0.9rem;">Heatmap Opacity: <span id="opacityValue">0.40</span></label>
                    <input type="range" id="heatmapOpacity" min="0" max="100" value="40" style="width: 100%; accent-color: #ff6b6b; margin-top: 5px;">
                </div>
            </div>
            <div class="images-grid">
                <div class="image-card" style="grid-column: 1 / -1;">
                    <h3>Heatmap</h3>
                    <canvas id="heatmapCanvas" style="width:100%; border-radius:10px;"></canvas>
                </div>
            </div>
            <div id="predictionsList" style="margin-top: 15px;"></div>
        </div>
        
        <footer>
            <p>OsTrace v2.0.0 — On-Prem Fracture Detection</p>
        </footer>
    </div>
    
    <script>
        let selectedFile = null;
        let isDicom = false;
        let lastPredictions = [];
        let originalImage = null; // Image object for the original
        let imgW = 0, imgH = 0;
        
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
            if (!file) { showError('Please select a file'); return; }
            const fileName = file.name.toLowerCase();
            isDicom = fileName.endsWith('.dcm') || fileName.endsWith('.dicom');
            if (!isDicom && !file.type.startsWith('image/')) {
                showError('Please select a valid image file or DICOM file');
                return;
            }
            selectedFile = file;
            if (isDicom) {
                document.getElementById('previewImage').src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMjIyIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZpbGw9IiM4ODgiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+RElDT00gRmlsZTwvdGV4dD48L3N2Zz4=';
            } else {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('previewImage').src = e.target.result;
                };
                reader.readAsDataURL(file);
            }
            document.getElementById('previewContainer').classList.add('active');
            document.getElementById('resultsSection').classList.remove('active');
            document.getElementById('errorMessage').classList.remove('active');
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('errorMessage');
            errorDiv.textContent = message;
            errorDiv.classList.add('active');
        }
        
        // JET colormap lookup (simplified 256 entries)
        function jetColor(t) {
            // t in [0,1], returns [r, g, b]
            let r, g, b;
            if (t < 0.125) { r = 0; g = 0; b = 0.5 + t * 4; }
            else if (t < 0.375) { r = 0; g = (t - 0.125) * 4; b = 1; }
            else if (t < 0.625) { r = (t - 0.375) * 4; g = 1; b = 1 - (t - 0.375) * 4; }
            else if (t < 0.875) { r = 1; g = 1 - (t - 0.625) * 4; b = 0; }
            else { r = 1 - (t - 0.875) * 4; g = 0; b = 0; }
            return [Math.round(Math.max(0, Math.min(1, r)) * 255),
                    Math.round(Math.max(0, Math.min(1, g)) * 255),
                    Math.round(Math.max(0, Math.min(1, b)) * 255)];
        }
        
        function renderHeatmap() {
            if (!originalImage) return;
            const canvas = document.getElementById('heatmapCanvas');
            const ctx = canvas.getContext('2d');
            canvas.width = imgW;
            canvas.height = imgH;
            
            const threshold = document.getElementById('confidenceThreshold').value / 100;
            const opacity = document.getElementById('heatmapOpacity').value / 100;
            
            // Draw original image
            ctx.drawImage(originalImage, 0, 0, imgW, imgH);
            
            // Filter predictions by threshold
            const filtered = lastPredictions.filter(p => (p.confidence || 0) >= threshold);
            
            // Update badge
            const badge = document.getElementById('countBadge');
            badge.textContent = filtered.length > 0 ? filtered.length + ' fracture(s) detected' : 'No fractures detected';
            
            if (filtered.length === 0 || opacity === 0) {
                renderPredictions(lastPredictions, threshold);
                return;
            }
            
            // Generate heatmap on offscreen canvas
            const heatCanvas = document.createElement('canvas');
            heatCanvas.width = imgW;
            heatCanvas.height = imgH;
            const hCtx = heatCanvas.getContext('2d');
            
            // Build heatmap array
            const heatData = new Float32Array(imgW * imgH);
            
            for (const p of filtered) {
                const cx = p.x || 0;
                const cy = p.y || 0;
                const bw = p.width || 0;
                const bh = p.height || 0;
                const conf = p.confidence || 0;
                const sigX = Math.max(bw * 0.6, 20);
                const sigY = Math.max(bh * 0.6, 20);
                
                // Only compute within 3 sigma
                const x1 = Math.max(0, Math.floor(cx - sigX * 3));
                const x2 = Math.min(imgW, Math.ceil(cx + sigX * 3));
                const y1 = Math.max(0, Math.floor(cy - sigY * 3));
                const y2 = Math.min(imgH, Math.ceil(cy + sigY * 3));
                
                for (let y = y1; y < y2; y++) {
                    for (let x = x1; x < x2; x++) {
                        const dx = (x - cx) / sigX;
                        const dy = (y - cy) / sigY;
                        const g = Math.exp(-0.5 * (dx * dx + dy * dy)) * conf;
                        const idx = y * imgW + x;
                        heatData[idx] += g;
                    }
                }
            }
            
            // Normalize
            let maxVal = 0;
            for (let i = 0; i < heatData.length; i++) {
                if (heatData[i] > maxVal) maxVal = heatData[i];
            }
            
            // Draw heatmap with JET colormap
            const imgData = hCtx.createImageData(imgW, imgH);
            for (let i = 0; i < heatData.length; i++) {
                const t = maxVal > 0 ? heatData[i] / maxVal : 0;
                const [r, g, b] = jetColor(t);
                imgData.data[i * 4] = r;
                imgData.data[i * 4 + 1] = g;
                imgData.data[i * 4 + 2] = b;
                imgData.data[i * 4 + 3] = 255;
            }
            hCtx.putImageData(imgData, 0, 0);
            
            // Blend heatmap over original
            ctx.globalAlpha = opacity;
            ctx.drawImage(heatCanvas, 0, 0);
            ctx.globalAlpha = 1.0;
            
            renderPredictions(lastPredictions, threshold);
        }
        
        document.getElementById('confidenceThreshold').addEventListener('input', function(e) {
            const val = (e.target.value / 100).toFixed(2);
            document.getElementById('thresholdValue').textContent = val;
            renderHeatmap();
        });
        
        document.getElementById('heatmapOpacity').addEventListener('input', function(e) {
            const val = (e.target.value / 100).toFixed(2);
            document.getElementById('opacityValue').textContent = val;
            renderHeatmap();
        });
        
        async function analyzeImage() {
            if (!selectedFile) { showError('Please select an image first'); return; }
            
            const btn = document.getElementById('analyzeBtn');
            const loading = document.getElementById('loading');
            const results = document.getElementById('resultsSection');
            const errorMessage = document.getElementById('errorMessage');
            
            btn.disabled = true;
            loading.classList.add('active');
            results.classList.remove('active');
            errorMessage.classList.remove('active');
            
            try {
                const formData = new FormData();
                formData.append('file', selectedFile);
                
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || errorData.detail || 'Server error');
                }
                
                const result = await response.json();
                
                // Load original image
                const img = new Image();
                img.onload = function() {
                    originalImage = img;
                    imgW = img.width;
                    imgH = img.height;
                    lastPredictions = result.predictions || [];
                    renderHeatmap();
                    results.classList.add('active');
                };
                let src = result.original_image || '';
                if (src && !src.startsWith('data:')) {
                    src = 'data:image/jpeg;base64,' + src;
                }
                img.src = src;
                
            } catch (error) {
                showError('Analysis failed: ' + error.message);
            } finally {
                btn.disabled = false;
                loading.classList.remove('active');
            }
        }
        
        function renderPredictions(preds, threshold) {
            const container = document.getElementById('predictionsList');
            const filtered = preds.filter(p => (p.confidence || 0) >= threshold);
            
            if (filtered.length === 0) {
                container.innerHTML = '<div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:15px; text-align:center; color:#888;">No predictions above threshold (' + threshold.toFixed(2) + ')</div>';
                return;
            }
            
            let html = '';
            filtered.forEach((p, i) => {
                const conf = ((p.confidence || 0) * 100).toFixed(1);
                const cls = p.class || 'fracture';
                const color = p.confidence >= 0.7 ? '#ff6b6b' : (p.confidence >= 0.4 ? '#ffa94d' : '#868e96');
                html += '<div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:12px 18px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center; border-left:3px solid ' + color + ';">';
                html += '<span style="color:#ddd;">#' + (i+1) + ' <strong>' + cls + '</strong></span>';
                html += '<span style="color:' + color + '; font-weight:bold;">' + conf + '%</span>';
                html += '</div>';
            });
            container.innerHTML = html;
        }
        
        async function checkHealth() {
            try {
                const resp = await fetch('/health');
                const data = await resp.json();
                const dot = document.getElementById('statusDot');
                const text = document.getElementById('statusText');
                if (data.detector_ready) {
                    dot.className = 'status-dot ok';
                    text.textContent = 'ONNX модель загружена — готово к анализу';
                    text.style.color = '#51cf66';
                } else {
                    dot.className = 'status-dot error';
                    text.textContent = 'Модель не найдена';
                    text.style.color = '#ff6b6b';
                    document.getElementById('analyzeBtn').disabled = true;
                }
            } catch(e) {
                document.getElementById('statusText').textContent = 'Ошибка подключения';
            }
        }
        
        checkHealth();
    </script>
</body>
</html>
"""


@webapp.get("/", response_class=HTMLResponse)
async def root():
    return HTML_TEMPLATE


@webapp.get("/health")
async def health():
    return {
        "status": "healthy",
        "detector_ready": offline_detector is not None,
        "dicom_support": HAS_PYDICOM,
    }


def generate_heatmap_from_bboxes(
    image: np.ndarray,
    predictions: list,
    threshold: float = 0.25,
) -> np.ndarray:
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for pred in predictions:
        conf = pred.get("confidence", 0)
        if conf < threshold:
            continue

        cx = pred.get("x", 0)
        cy = pred.get("y", 0)
        bw = pred.get("width", 0)
        bh = pred.get("height", 0)

        sigma_x = max(bw * 0.6, 20)
        sigma_y = max(bh * 0.6, 20)

        y_coords, x_coords = np.mgrid[0:h, 0:w]
        gaussian = np.exp(
            -((x_coords - cx) ** 2 / (2 * sigma_x ** 2)
              + (y_coords - cy) ** 2 / (2 * sigma_y ** 2))
        )
        heatmap += gaussian * conf

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    if image.max() <= 1.0:
        vis_image = (image * 255).astype(np.uint8)
    else:
        vis_image = image.astype(np.uint8)

    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    elif vis_image.shape[2] == 3:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

    alpha = 0.4
    overlay = cv2.addWeighted(vis_image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


@webapp.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    if offline_detector is None:
        return JSONResponse(
            {"error": "ONNX model not loaded. Place weights.onnx in exported_models_v5/"},
            status_code=503,
        )

    try:
        image_bytes = await file.read()
        filename = file.filename.lower() if file.filename else ""
        is_dicom_file = filename.endswith('.dcm') or filename.endswith('.dicom')

        if is_dicom_file:
            if not HAS_PYDICOM:
                return JSONResponse(
                    {"error": "DICOM support not available. Install pydicom: pip install pydicom"},
                    status_code=400,
                )
            image_array = load_dicom(image_bytes)
        else:
            image_array = np.array(
                Image.open(BytesIO(image_bytes)).convert("RGB"),
                dtype=np.float32,
            ) / 255.0

        result = offline_detector.predict(image_array)
        preds = result.get("predictions", [])
        count = result.get("count_objects", 0)

        original_base64 = encode_image_to_base64(image_array)

        return {
            "count_objects": count,
            "predictions": preds,
            "original_image": original_base64,
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=400)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(webapp, host="0.0.0.0", port=8080)
