from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cloudinary
import cloudinary.uploader

from .config import (
    CLOUDINARY_API_KEY,
    CLOUDINARY_API_SECRET,
    CLOUDINARY_CLOUD_NAME,
    CLOUDINARY_ENABLED,
    CLOUDINARY_FOLDER,
    CLOUDINARY_TTL_DAYS,
)

_configured = False


def _is_configured() -> bool:
    if not CLOUDINARY_ENABLED:
        return False
    return bool(CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET)


def is_cloudinary_ready() -> bool:
    return _configured and _is_configured()


def init_cloudinary() -> bool:
    global _configured
    if _configured:
        return True
    if not _is_configured():
        return False
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True,
    )
    _configured = True
    return True


def upload_page_image(local_path: Path, public_id: str, job_id: str) -> Optional[str]:
    if not init_cloudinary():
        return None

    expire_at = datetime.now(timezone.utc) + timedelta(days=CLOUDINARY_TTL_DAYS)
    tags = ["nha-docguard", f"job:{job_id}", f"expire:{expire_at.date().isoformat()}"]
    context = {"expire_at": expire_at.isoformat()}

    try:
        result = cloudinary.uploader.upload(
            str(local_path),
            folder=CLOUDINARY_FOLDER,
            public_id=public_id,
            overwrite=True,
            unique_filename=False,
            resource_type="image",
            tags=tags,
            context=context,
        )
    except Exception:
        return None

    return result.get("secure_url") or result.get("url")


def _resource_type_for_path(local_path: Path) -> str:
    suffix = local_path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
        return "image"
    return "raw"


def upload_job_file(local_path: Path, public_id: str, job_id: str) -> Optional[str]:
    if not init_cloudinary():
        return None

    expire_at = datetime.now(timezone.utc) + timedelta(days=CLOUDINARY_TTL_DAYS)
    tags = ["nha-docguard", f"job:{job_id}", f"expire:{expire_at.date().isoformat()}"]
    context = {"expire_at": expire_at.isoformat()}
    resource_type = _resource_type_for_path(local_path)
    scoped_public_id = f"{job_id}/{public_id}"

    try:
        result = cloudinary.uploader.upload(
            str(local_path),
            folder=CLOUDINARY_FOLDER,
            public_id=scoped_public_id,
            overwrite=True,
            unique_filename=False,
            resource_type=resource_type,
            tags=tags,
            context=context,
        )
    except Exception:
        return None

    return result.get("secure_url") or result.get("url")
