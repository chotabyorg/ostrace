from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

import handler
import webapp


class FakeDetector:
    def predict(self, image_array):
        height, width = image_array.shape[:2]
        return {
            "count_objects": 1,
            "predictions": [
                {
                    "x": width / 2,
                    "y": height / 2,
                    "width": width / 4,
                    "height": height / 4,
                    "confidence": 0.9,
                    "class": "fracture",
                    "class_id": 1,
                }
            ],
        }


def make_png_bytes():
    image = Image.new("RGB", (8, 8), color=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def assert_predict_contract(payload):
    assert set(payload) == {"count_objects", "predictions", "original_image"}
    assert payload["count_objects"] == 1
    assert payload["predictions"][0]["class"] == "fracture"
    assert payload["original_image"]


def test_webapp_predict_contract(monkeypatch):
    monkeypatch.setattr(webapp, "offline_detector", FakeDetector())
    client = TestClient(webapp.webapp)

    response = client.post(
        "/api/predict",
        files={"file": ("xray.png", make_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert_predict_contract(response.json())


def test_handler_predict_contract(monkeypatch):
    monkeypatch.setattr(handler, "detector", FakeDetector())
    client = TestClient(handler.app)

    response = client.post(
        "/api/predict",
        files={"file": ("xray.png", make_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert_predict_contract(response.json())


def test_webapp_rejects_unsupported_file_type(monkeypatch):
    monkeypatch.setattr(webapp, "offline_detector", FakeDetector())
    client = TestClient(webapp.webapp)

    response = client.post(
        "/api/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
