# OsTrace — Обнаружение и локализация переломов

Многозадачная модель глубокого обучения для обнаружения и локализации костных аномалий на рентгеновских снимках.

## Возможности

- Классификация переломов (бинарная классификация)
- Локализация переломов (маска сегментации)
- Визуализация внимания Grad-CAM
- EfficientNetV2 backbone с механизмами внимания
- REST API для инференса
- Веб-интерфейс для тестирования
- Функционал загрузки модели

## Установка

```bash
pip install -r requirements.txt
```

## Использование

### Python API

```python
from ostrace import FractureDetector, Config

# Загрузка модели
config = Config(model_path="path/to/model.keras")
detector = FractureDetector(config)

# Предсказание
result = detector.predict("xray_image.jpg")
print(f"Перелом обнаружен: {result['has_fracture']}")
print(f"Уверенность: {result['confidence']}")
```

### REST API

Запуск сервера:

```bash
# Только API
python -m ostrace.handler

# или через uvicorn
uvicorn ostrace.handler:app --host 0.0.0.0 --port 8000
```

### Веб-интерфейс(ПЛЕЙСХОЛДЕР!)

Запуск веб-интерфейса:

```bash
# Веб-интерфейс
python -m ostrace.webapp

# или через uvicorn
uvicorn ostrace.webapp:webapp --host 0.0.0.0 --port 8080
```

Открывать через http://localhost:8080 в браузере.

## Эндпоинты

### POST /models/upload

Загрузка файла модели.

```bash
curl -X POST -F "file=@model.keras" -F "backbone=efficientnetv2-b3" http://localhost:8000/models/upload
```

Ответ:
```json
{
  "status": "success",
  "current_model": "model.keras",
  "config": {
    "image_size": 224,
    "mask_size": 224,
    "backbone": "efficientnetv2-b3"
  }
}
```

### POST /predict

Предсказание перелома по изображению в base64.

```bash
curl -X POST -F "image=<base64_image>" http://localhost:8000/predict
```

Ответ:
```json
{
  "has_fracture": true,
  "confidence": 0.87,
  "processed_image": "<base64>",
  "gradcam_image": "<base64>"
}
```

### POST /predict/upload

Предсказание по загруженному файлу.

```bash
curl -X POST -F "file=@xray.jpg" http://localhost:8000/predict/upload
```

### GET /health

Проверка состояния API.

```bash
curl http://localhost:8000/health
```

Ответ:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "current_model": "model.keras"
}
```

## Ссылка на гдрайв с моделью
```json
https://drive.google.com/drive/folders/1Ca2TZxPPA1fqemfrkBNGnZc9nEtEmqX5?usp=sharing
```
## Требования

- Python 3.9+
- TensorFlow 2.16+
- Keras 3.10+
- FastAPI
- OpenCV
- Pillow
- pydantic 
- pydicom





