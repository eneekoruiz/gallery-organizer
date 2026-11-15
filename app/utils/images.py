"""Decode gallery images to BGR, including iPhone HEIF/HEIC originals."""
from io import BytesIO

import cv2
import numpy as np


def decode_image(data):
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        encoded = data
    else:
        try:
            encoded = np.frombuffer(data, dtype=np.uint8)
        except (TypeError, ValueError):
            return None
    if not encoded.size:
        return None
    try:
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (cv2.error, Exception):
        image = None
    if image is not None:
        return image
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener(thumbnails=False)
    except Exception:
        pass
    try:
        raw_bytes = data if isinstance(data, (bytes, bytearray)) else encoded.tobytes()
        with Image.open(BytesIO(raw_bytes)) as photo:
            return cv2.cvtColor(np.asarray(ImageOps.exif_transpose(photo).convert('RGB')), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, ValueError, Exception):
        return None
