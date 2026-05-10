from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

import safety
from safety import classify_upload, decode_image_bytes, read_limited_upload


def make_png_bytes(size=(2, 2)):
    image = Image.new("RGB", size, color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_read_limited_upload_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(safety, "MAX_UPLOAD_BYTES", 4)
    upload = UploadFile(filename="xray.png", file=BytesIO(b"12345"))

    with pytest.raises(HTTPException) as exc_info:
        await read_limited_upload(upload)

    assert exc_info.value.status_code == 413


def test_classify_upload_accepts_supported_types():
    assert classify_upload("xray.jpg", "image/jpeg") == "image"
    assert classify_upload("scan.dcm", "application/octet-stream") == "dicom"


def test_classify_upload_rejects_unknown_types():
    with pytest.raises(HTTPException) as exc_info:
        classify_upload("notes.txt", "text/plain")

    assert exc_info.value.status_code == 415


def test_decode_image_bytes_returns_normalized_rgb_array():
    image = decode_image_bytes(make_png_bytes())

    assert image.shape == (2, 2, 3)
    assert image.dtype.name == "float32"
    assert image.max() <= 1.0
