import os
import warnings
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image

MAX_UPLOAD_BYTES = int(os.getenv("OSTRACE_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("OSTRACE_MAX_IMAGE_PIXELS", str(25_000_000)))
UPLOAD_CHUNK_BYTES = 1024 * 1024

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DICOM_EXTENSIONS = {".dcm", ".dicom"}

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def get_cors_origins() -> list[str]:
    raw_origins = os.getenv(
        "OSTRACE_CORS_ORIGINS",
        "http://127.0.0.1:8080,http://localhost:8080",
    )
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


async def read_limited_upload(file: UploadFile) -> bytes:
    chunks = []
    total_size = 0

    while True:
        chunk = await file.read(UPLOAD_CHUNK_BYTES)
        if not chunk:
            break

        total_size += len(chunk)
        if total_size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File is too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
        chunks.append(chunk)

    if total_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return b"".join(chunks)


def classify_upload(filename: Optional[str], content_type: Optional[str] = None) -> Literal["image", "dicom"]:
    extension = Path(filename or "").suffix.lower()
    content_type = (content_type or "").lower()

    if extension in DICOM_EXTENSIONS:
        return "dicom"
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if content_type.startswith("image/") and content_type not in {"image/svg+xml"}:
        return "image"

    raise HTTPException(
        status_code=415,
        detail="Unsupported file type. Upload JPG, PNG, DICOM (.dcm), or DICOM (.dicom).",
    )


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                image.verify()

            with Image.open(BytesIO(image_bytes)) as image:
                pixel_count = image.width * image.height
                if pixel_count > MAX_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image is too large. Maximum pixel count is {MAX_IMAGE_PIXELS}.",
                    )
                return np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large. Maximum pixel count is {MAX_IMAGE_PIXELS}.",
        ) from None
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.") from None


def validate_dicom_dimensions(image_bytes: bytes) -> None:
    try:
        import pydicom

        metadata = pydicom.dcmread(BytesIO(image_bytes), stop_before_pixels=True)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable DICOM file.") from None

    try:
        rows = int(getattr(metadata, "Rows", 0) or 0)
        columns = int(getattr(metadata, "Columns", 0) or 0)
        samples = int(getattr(metadata, "SamplesPerPixel", 1) or 1)
        frames = int(getattr(metadata, "NumberOfFrames", 1) or 1)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="DICOM file has invalid image dimensions.") from None

    pixel_count = rows * columns * samples * frames
    if pixel_count <= 0:
        raise HTTPException(status_code=400, detail="DICOM file has missing image dimensions.")
    if pixel_count > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"DICOM image is too large. Maximum pixel count is {MAX_IMAGE_PIXELS}.",
        )
