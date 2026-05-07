import mimetypes
from datetime import timedelta
from pathlib import Path
from typing import Optional

from .config import SIGNED_URL_DAYS
from .firebase import get_storage_bucket


def upload_file(local_path: Path, dest_path: str, content_type: Optional[str] = None) -> Optional[str]:
    bucket = get_storage_bucket()
    if bucket is None:
        return None

    blob = bucket.blob(dest_path)
    ct = content_type
    if ct is None:
        ct, _ = mimetypes.guess_type(str(local_path))
    blob.upload_from_filename(str(local_path), content_type=ct or "application/octet-stream")

    try:
        return blob.generate_signed_url(expiration=timedelta(days=SIGNED_URL_DAYS), method="GET")
    except Exception:
        return None
