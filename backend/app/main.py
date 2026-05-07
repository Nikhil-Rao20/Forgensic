import json
import os
import uuid
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import (
    AUTH_REQUIRED,
    CORS_ORIGINS,
    DATA_DIR,
    JOB_EXECUTOR_WORKERS,
    MAX_UPLOAD_BYTES,
    PIPELINE_PRESET,
    PIPELINE_VERSION,
)
from .cloudinary_uploader import init_cloudinary, upload_job_file
from .firebase import get_firestore, verify_id_token
from .models import JobCreateResponse, JobResultResponse, JobStatusResponse
from .pipeline import DetectedRegion, DocumentPage, PageAnalysisResult, run_pipeline

app = FastAPI(title="NHA PS3 Forensics API")

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

DATA_DIR.mkdir(parents=True, exist_ok=True)

_executor = ThreadPoolExecutor(max_workers=JOB_EXECUTOR_WORKERS)
_JOB_STATE: Dict[str, Dict[str, Any]] = {}
_JOB_RESULTS: Dict[str, Dict[str, Any]] = {}
_JOB_FILES: Dict[str, Dict[str, str]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_user_info(authorization: Optional[str]) -> Optional[Dict[str, Any]]:
    if not authorization:
        return None
    if not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        return None
    decoded = verify_id_token(token)
    if not decoded:
        return None
    return {
        "uid": decoded.get("uid"),
        "email": decoded.get("email"),
        "name": decoded.get("name") or decoded.get("displayName"),
    }


def _require_user_info(authorization: Optional[str]) -> Dict[str, Any]:
    info = _get_user_info(authorization)
    if AUTH_REQUIRED and not info:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return info or {"uid": "anonymous"}


def _write_job_state(job_id: str, payload: Dict[str, Any]) -> None:
    _JOB_STATE.setdefault(job_id, {}).update(payload)
    db = get_firestore()
    if db is None:
        return
    db.collection("jobs").document(job_id).set(payload, merge=True)


def _save_upload(upload: UploadFile, dest: Path) -> int:
    size = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File exceeds 10MB limit")
            f.write(chunk)
    return size


def _allowed_suffix(name: str) -> bool:
    suffix = Path(name).suffix.lower()
    return suffix in {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _region_to_dict(region: DetectedRegion) -> Dict[str, Any]:
    return {
        "x": region.x,
        "y": region.y,
        "w": region.w,
        "h": region.h,
        "category_id": region.category_id,
        "type": region.type,
        "stretch_factor": region.stretch_factor,
        "header_source": region.header_source,
        "body_source": region.body_source,
    }


def _result_to_dict(result: PageAnalysisResult, page: Optional[DocumentPage], image_url: Optional[str]) -> Dict[str, Any]:
    return {
        "page_id": f"{result.file_name}",
        "page_number": result.page_number,
        "file_name": result.file_name,
        "image_url": image_url,
        "image_width": page.image_width if page else None,
        "image_height": page.image_height if page else None,
        "categories": result.predicted_categories,
        "regions": [_region_to_dict(r) for r in result.detected_regions],
        "notes": result.notes,
    }


def _build_results_payload(
    job_id: str,
    file_name: str,
    pages: list,
    results: list,
    export_info: Dict[str, Any],
    file_url_map: Dict[str, str],
    inference_seconds: Optional[float] = None,
    avg_inference_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    page_map = {p.page_file_name: p for p in pages}
    payload_pages = []
    summary: Dict[str, int] = {}
    for res in results:
        page = page_map.get(res.file_name)
        image_url = file_url_map.get(res.file_name)
        payload_pages.append(_result_to_dict(res, page, image_url))
        for cat in res.predicted_categories:
            summary[cat] = summary.get(cat, 0) + 1

    export_urls = {
        "json": file_url_map.get("submission.json"),
        "excel": file_url_map.get("submission_preview.xlsx"),
        "yaml": [file_url_map.get(Path(p).name) for p in export_info.get("yaml_paths", []) if file_url_map.get(Path(p).name)],
    }

    return {
        "job_id": job_id,
        "status": "complete",
        "file_name": file_name,
        "pipeline_version": PIPELINE_VERSION,
        "pages": payload_pages,
        "category_summary": summary,
        "export_urls": export_urls,
        "inference_seconds": inference_seconds,
        "avg_inference_seconds": avg_inference_seconds,
        "created_at": _JOB_STATE.get(job_id, {}).get("created_at"),
        "updated_at": _JOB_STATE.get(job_id, {}).get("updated_at"),
    }


def _register_job_file(job_id: str, name: str, path: Path) -> None:
    _JOB_FILES.setdefault(job_id, {})[name] = str(path)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except Exception:
        return


def _process_job(job_id: str, user_id: str, file_path: Path, preset: str) -> None:
    _write_job_state(job_id, {"status": "processing", "updated_at": _now_iso(), "progress": 0.1})
    job_dir = DATA_DIR / job_id
    cloudinary_enabled = init_cloudinary()
    all_cloud_uploaded = cloudinary_enabled

    try:
        inference_start = perf_counter()
        run_output = run_pipeline(file_path, job_dir, preset=preset)
        inference_seconds = perf_counter() - inference_start
        pages = run_output["pages"]
        results = run_output["results"]
        export_info = run_output["export_info"]

        avg_inference_seconds = None
        if pages:
            avg_inference_seconds = inference_seconds / max(len(pages), 1)

        file_url_map: Dict[str, str] = {}

        def _store_file(file_name: str, local_path: Path) -> Optional[str]:
            nonlocal all_cloud_uploaded
            local_url = f"/jobs/{job_id}/files/{file_name}"
            cloud_url = None
            if cloudinary_enabled:
                cloud_url = upload_job_file(local_path, file_name, job_id)
            if cloud_url:
                file_url_map[file_name] = cloud_url
                _safe_unlink(local_path)
                return cloud_url
            if cloudinary_enabled:
                all_cloud_uploaded = False
            _register_job_file(job_id, file_name, local_path)
            file_url_map[file_name] = local_url
            return local_url

        # Register rendered pages and output files for local serving
        for page in pages:
            if page.image_path:
                _store_file(page.page_file_name, Path(page.image_path))

        output_dir = Path(run_output["output_dir"])
        annotations_dir = Path(run_output["annotations_dir"])
        submission_json = output_dir / "submission.json"
        if submission_json.exists():
            _store_file(submission_json.name, submission_json)

        excel_preview = output_dir / "submission_preview.xlsx"
        if excel_preview.exists():
            _store_file(excel_preview.name, excel_preview)

        for yaml_path in annotations_dir.glob("*.yaml"):
            _store_file(yaml_path.name, yaml_path)

        payload = _build_results_payload(
            job_id,
            file_path.name,
            pages,
            results,
            export_info,
            file_url_map,
            inference_seconds,
            avg_inference_seconds,
        )
        _JOB_RESULTS[job_id] = payload

        _write_job_state(
            job_id,
            {
                "status": "complete",
                "updated_at": _now_iso(),
                "progress": 1.0,
                "inference_seconds": inference_seconds,
                "avg_inference_seconds": avg_inference_seconds,
                "result": payload,
            },
        )

        input_url = _store_file(file_path.name, file_path)
        if input_url:
            _write_job_state(job_id, {"input_url": input_url})
            payload["input_url"] = input_url

        if cloudinary_enabled and all_cloud_uploaded:
            shutil.rmtree(job_dir, ignore_errors=True)

    except Exception as exc:
        _write_job_state(job_id, {"status": "error", "updated_at": _now_iso(), "message": str(exc)})


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "time": _now_iso()}


@app.post("/jobs", response_model=JobCreateResponse)
async def create_job(request: Request, file: UploadFile = File(...)) -> JobCreateResponse:
    user = _require_user_info(request.headers.get("authorization"))
    user_id = user.get("uid")
    user_email = user.get("email")
    user_name = user.get("name")

    if not file.filename or not _allowed_suffix(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    job_id = uuid.uuid4().hex
    job_dir = DATA_DIR / job_id
    input_path = job_dir / "input" / file.filename
    size = _save_upload(file, input_path)

    _write_job_state(
        job_id,
        {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "file_name": file.filename,
            "file_size": size,
            "user_id": user_id,
            "user_email": user_email,
            "user_name": user_name,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "pipeline_version": PIPELINE_VERSION,
        },
    )

    _executor.submit(_process_job, job_id, user_id, input_path, PIPELINE_PRESET)

    return JobCreateResponse(job_id=job_id, status="queued", message="Job accepted")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, request: Request) -> JobStatusResponse:
    _require_user_info(request.headers.get("authorization"))

    state = _JOB_STATE.get(job_id)
    if not state:
        db = get_firestore()
        if db:
            doc = db.collection("jobs").document(job_id).get()
            if doc.exists:
                state = doc.to_dict()
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job_id,
        status=state.get("status", "unknown"),
        progress=state.get("progress"),
        message=state.get("message"),
        created_at=state.get("created_at"),
        updated_at=state.get("updated_at"),
    )


@app.get("/jobs/{job_id}/results", response_model=JobResultResponse)
async def get_job_results(job_id: str, request: Request) -> JobResultResponse:
    _require_user_info(request.headers.get("authorization"))

    payload = _JOB_RESULTS.get(job_id)
    if not payload:
        db = get_firestore()
        if db:
            doc = db.collection("jobs").document(job_id).get()
            if doc.exists:
                data = doc.to_dict()
                payload = data.get("result")
    if not payload:
        raise HTTPException(status_code=404, detail="Results not ready")

    return JobResultResponse(**payload)


@app.get("/jobs/{job_id}/files/{file_name}")
async def get_job_file(job_id: str, file_name: str, request: Request):
    _require_user_info(request.headers.get("authorization"))

    job_files = _JOB_FILES.get(job_id, {})
    path_str = job_files.get(file_name)
    if not path_str:
        raise HTTPException(status_code=404, detail="File not found")
    path = Path(path_str)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)
