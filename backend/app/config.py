import os
from pathlib import Path

APP_ENV = os.getenv("APP_ENV", "prod")
DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/ps3-data"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]

FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS")
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET")
AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "true").lower() == "true"

PIPELINE_PRESET = os.getenv("PIPELINE_PRESET", "super_loose")
PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "ps3-cv-1.0.0")
JOB_EXECUTOR_WORKERS = int(os.getenv("JOB_EXECUTOR_WORKERS", "1"))
SIGNED_URL_DAYS = int(os.getenv("SIGNED_URL_DAYS", "7"))

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER", "nha-docguard")
CLOUDINARY_TTL_DAYS = int(os.getenv("CLOUDINARY_TTL_DAYS", "7"))
