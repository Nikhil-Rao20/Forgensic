from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class RegionModel(BaseModel):
    x: int
    y: int
    w: int
    h: int
    category_id: str
    type: Optional[str] = None
    stretch_factor: Optional[float] = None
    header_source: Optional[str] = None
    body_source: Optional[str] = None


class PageResultModel(BaseModel):
    page_id: str
    page_number: int
    file_name: str
    image_url: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    categories: List[str]
    regions: List[RegionModel]
    notes: Dict[str, Any]


class JobResultResponse(BaseModel):
    job_id: str
    status: str
    file_name: str
    pipeline_version: str
    pages: List[PageResultModel]
    category_summary: Dict[str, int]
    export_urls: Dict[str, Any]
    inference_seconds: Optional[float] = None
    avg_inference_seconds: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
