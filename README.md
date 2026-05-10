# OsTrace

OsTrace - локальный инструмент для обнаружения и визуализации возможных переломов на рентгеновских снимках. Проект работает с ONNX-моделью, умеет принимать обычные изображения и DICOM-файлы, предоставляет веб-интерфейс, FastAPI API и отдельное desktop-приложение на CustomTkinter.

Важно: результат модели не является медицинским диагнозом. Используйте его только как вспомогательный инструмент.

## Где скачать модель

Модель не хранится в репозитории, потому что файл весит слишком много. Скачайте ONNX-модель из Google Drive:

**[Скачать модель OsTrace из Google Drive](https://drive.google.com/drive/folders/1Ca2TZxPPA1fqemfrkBNGnZc9nEtEmqX5?usp=sharing)**

Прямая ссылка, если кнопка выше не открывается:

```text
https://drive.google.com/drive/folders/1Ca2TZxPPA1fqemfrkBNGnZc9nEtEmqX5?usp=sharing
```

После скачивания положите файл `.onnx` в папку `ostracemodel/` в корне проекта.

## Что есть в проекте

- `webapp.py` - веб-интерфейс и API `/api/predict`.
- `handler.py` - минимальный FastAPI API без HTML-интерфейса.
- `app.py` - standalone desktop UI.
- `inference.py` - загрузка ONNX-модели, preprocess, postprocess и NMS.
- `ostracemodel/` - ожидаемая папка для ONNX-модели. Ее нужно создать вручную, если ее нет.

## Требования

- Python 3.9-3.12.
- ONNX-модель OsTrace в формате `.onnx`.
- Для DICOM-файлов нужен `pydicom` - он уже указан в `requirements.txt`.

На Python 3.13+ часть ML-зависимостей может быть недоступна или работать нестабильно, поэтому для обычной установки лучше использовать Python 3.10-3.12.

## Быстрый старт

### 1. Склонировать репозиторий

```bash
git clone https://github.com/chotabyorg/ostrace.git
cd ostrace
```

### 2. Создать виртуальное окружение

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Установить зависимости

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Добавить модель

Скачайте ONNX-модель из Google Drive:

**[Скачать модель OsTrace из Google Drive](https://drive.google.com/drive/folders/1Ca2TZxPPA1fqemfrkBNGnZc9nEtEmqX5?usp=sharing)**

Затем создайте папку `ostracemodel` в корне проекта и положите туда скачанный `.onnx` файл:

```bash
mkdir -p ostracemodel
```

Пример структуры:

```text
ostrace/
  ostracemodel/
    model.onnx
  webapp.py
  inference.py
```

Имя ONNX-файла может быть любым, но в папке лучше держать одну актуальную модель, потому что приложение загружает первый найденный `.onnx` файл.

## Запуск веб-интерфейса

```bash
python -m uvicorn webapp:webapp --host 127.0.0.1 --port 8080 --reload
```

Откройте в браузере:

```text
http://127.0.0.1:8080
```

В интерфейсе можно загрузить `.jpg`, `.jpeg`, `.png`, `.dcm` или `.dicom` файл. Если модель не найдена, сверху будет статус "Модель не найдена", а анализ будет недоступен.

## Запуск минимального API

Если HTML-интерфейс не нужен:

```bash
python -m uvicorn handler:app --host 127.0.0.1 --port 8080 --reload
```

Проверка состояния:

```bash
curl http://127.0.0.1:8080/health
```

## API веб-приложения

### `GET /health`

Возвращает состояние веб-приложения:

```json
{
  "status": "healthy",
  "detector_ready": true,
  "dicom_support": true
}
```

`detector_ready: false` означает, что `.onnx` модель не найдена или не загрузилась.

### `POST /api/predict`

Принимает файл снимка через multipart form-data:

```bash
curl -X POST \
  -F "file=@xray.jpg" \
  http://127.0.0.1:8080/api/predict
```

Пример ответа:

```json
{
  "count_objects": 1,
  "predictions": [
    {
      "x": 230.5,
      "y": 180.2,
      "width": 75.0,
      "height": 62.0,
      "confidence": 0.87,
      "class": "fracture",
      "class_id": 1
    }
  ],
  "original_image": "<base64>"
}
```

## Запуск desktop-приложения

```bash
python app.py
```

Desktop UI автоматически ищет `.onnx` модель в папках `ostracemodel/` и `models/`. Также модель можно выбрать вручную кнопкой "Загрузить модель".

## Частые проблемы

### `ONNX model not loaded`

Проверьте, что файл модели лежит здесь:

```text
ostrace/ostracemodel/*.onnx
```

После добавления модели перезапустите сервер.

### `onnxruntime не установлен`

Установите зависимости заново в активированном виртуальном окружении:

```bash
python -m pip install -r requirements.txt
```

### Не открываются DICOM-файлы

Проверьте, что установлен `pydicom`:

```bash
python -m pip install pydicom
```

### Порт 8080 занят

Запустите на другом порту:

```bash
python -m uvicorn webapp:webapp --host 127.0.0.1 --port 8090 --reload
```

И откройте:

```text
http://127.0.0.1:8090
```

## Команды для разработки

Если нужно запускать тесты, установите dev-зависимости:

```bash
python -m pip install -e ".[dev]"
```

```bash
# Проверить синтаксис Python-файлов
python -m compileall .

# Запустить тесты
python -m pytest

# Запустить веб-интерфейс
python -m uvicorn webapp:webapp --host 127.0.0.1 --port 8080 --reload

# Запустить минимальный API
python -m uvicorn handler:app --host 127.0.0.1 --port 8080 --reload

# Запустить desktop UI
python app.py
```

## Примечания по модели

`inference.py` ожидает, что ONNX-модель возвращает:

1. bbox-координаты в нормализованном формате `cx, cy, width, height`;
2. logits/score-выход для классов.

Код поддерживает бинарный и многоклассовый score-выход. Если ваша модель экспортирована с другим форматом выходов, нужно адаптировать `postprocess()` в `inference.py`.
