# Backend (Render)

This FastAPI service runs the PS3 CV pipeline and exposes job-based APIs.

## Environment variables
- `FIREBASE_CREDENTIALS`: JSON string for Firebase service account.
- `FIREBASE_STORAGE_BUCKET`: Firebase storage bucket name.
- `CORS_ORIGINS`: Comma-separated list of allowed origins (GitHub Pages URL).
- `AUTH_REQUIRED`: `true` or `false`.
- `PIPELINE_PRESET`: `super_loose` by default.
- `PIPELINE_VERSION`: label shown in UI.
- `MAX_UPLOAD_BYTES`: default 10MB.
- `JOB_EXECUTOR_WORKERS`: number of worker threads.

## Cloudinary (optional)
- `CLOUDINARY_CLOUD_NAME`
- `CLOUDINARY_API_KEY`
- `CLOUDINARY_API_SECRET`
- `CLOUDINARY_FOLDER` (default: nha-docguard)
- `CLOUDINARY_TTL_DAYS` (default: 7)

## Run locally
```
uvicorn app.main:app --reload
```
