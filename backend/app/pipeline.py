import difflib
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import fitz
except ImportError:
    fitz = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

OCR_ENABLED = True


def set_ocr_enabled(enabled: bool) -> None:
    global OCR_ENABLED
    OCR_ENABLED = bool(enabled)


def is_ocr_enabled() -> bool:
    return OCR_ENABLED


CATEGORY_IDS = {
    "C1": "Copy-paste within the same document",
    "C2": "Overwriting on existing text",
    "C3": "Adding new content",
    "C4": "Removing / erasing text or image",
    "C5": "Merging content from different documents",
    "C6": "Watermark removal",
    "C7": "Irregular spacing",
    "C8": "Fully AI-generated document",
    "C9": "Partial AI-generated edits",
    "C10": "No editing / discrepancy found",
}

CATEGORY_ONLY_CLASSES = {"C8", "C10"}


@dataclass
class DocumentPage:
    source_link: str
    original_path: str
    original_file_name: str
    page_file_name: str
    page_number: int
    is_pdf: bool
    image_path: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None


@dataclass
class DetectedRegion:
    x: int
    y: int
    w: int
    h: int
    category_id: str
    type: Optional[str] = None
    stretch_factor: Optional[float] = None
    header_source: Optional[str] = None
    body_source: Optional[str] = None


@dataclass
class PageAnalysisResult:
    source_link: str
    file_name: str
    original_file_name: str
    page_number: int
    predicted_categories: List[str] = field(default_factory=list)
    detected_regions: List[DetectedRegion] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)


# =========================
# DATA ONBOARDING HELPERS
# =========================

def safe_open_image_size(image_path: Path) -> Tuple[Optional[int], Optional[int]]:
    if Image is None:
        return None, None
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return None, None


def render_pdf_to_images(pdf_path: Path, render_dir: Path) -> List[DocumentPage]:
    pages: List[DocumentPage] = []

    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is required to render PDF pages.")

    doc = fitz.open(pdf_path)
    try:
        for idx in range(len(doc)):
            page = doc.load_page(idx)
            pix = page.get_pixmap()
            page_file_name = f"{pdf_path.stem}_page_{idx + 1}.jpg"
            image_path = render_dir / page_file_name
            pix.save(str(image_path))

            width, height = safe_open_image_size(image_path)
            pages.append(
                DocumentPage(
                    source_link=str(pdf_path),
                    original_path=str(pdf_path),
                    original_file_name=pdf_path.name,
                    page_file_name=page_file_name,
                    page_number=idx + 1,
                    is_pdf=True,
                    image_path=str(image_path),
                    image_width=width,
                    image_height=height,
                )
            )
    finally:
        doc.close()

    return pages


def build_document_pages(input_dir: Path, render_dir: Path) -> List[DocumentPage]:
    pages: List[DocumentPage] = []
    render_dir.mkdir(parents=True, exist_ok=True)

    supported_image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    supported_pdf_exts = {".pdf"}

    for file_path in sorted(input_dir.iterdir()):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix in supported_image_exts:
            width, height = safe_open_image_size(file_path)
            pages.append(
                DocumentPage(
                    source_link=str(file_path),
                    original_path=str(file_path),
                    original_file_name=file_path.name,
                    page_file_name=file_path.name,
                    page_number=1,
                    is_pdf=False,
                    image_path=str(file_path),
                    image_width=width,
                    image_height=height,
                )
            )
        elif suffix in supported_pdf_exts:
            pages.extend(render_pdf_to_images(file_path, render_dir))

    return pages


# =========================
# PIPELINE STAGES
# =========================

def _load_page_gray(page: DocumentPage) -> Optional[np.ndarray]:
    if Image is None:
        return None
    image_path = page.image_path or page.original_path
    if image_path is None:
        return None
    try:
        with Image.open(image_path) as img:
            img = img.convert("L")
            return np.array(img)
    except Exception:
        return None


def _resize_max(image: np.ndarray, max_dim: int = 1200) -> Tuple[np.ndarray, float]:
    if image is None:
        return image, 1.0
    h, w = image.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale == 1.0:
        return image, 1.0
    if cv2 is None:
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        return np.array(Image.fromarray(image).resize((new_w, new_h))), scale
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), scale


def _text_mask(gray: np.ndarray) -> Optional[np.ndarray]:
    if gray is None:
        return None
    if cv2 is None:
        thresh = gray < np.mean(gray)
        return (thresh.astype(np.uint8) * 255)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _robust_z(value: float, values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad < 1e-6:
        return 0.0
    return 0.6745 * (value - med) / mad


def _extract_text_lines(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    if mask is None:
        return []
    row_density = (mask > 0).mean(axis=1)
    active = row_density > 0.02
    lines = []
    start = None
    for idx, val in enumerate(active):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            end = idx - 1
            lines.append((start, end))
            start = None
    if start is not None:
        lines.append((start, len(active) - 1))

    boxes = []
    for y1, y2 in lines:
        cols = (mask[y1 : y2 + 1, :] > 0).mean(axis=0)
        if cols.max() <= 0:
            continue
        xs = np.where(cols > 0.01)[0]
        if xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        if (y2 - y1) < 6 or (x2 - x1) < 20:
            continue
        boxes.append((x1, y1, x2 + 1, y2 + 1))
    return boxes


def _component_boxes(mask: np.ndarray, min_area: int = 120) -> List[Tuple[int, int, int, int]]:
    if cv2 is None:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def _component_stats(mask: np.ndarray, min_area: int = 200) -> List[Dict[str, Any]]:
    if cv2 is None:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        perim = float(cv2.arcLength(cnt, True))
        box_area = float(max(w * h, 1))
        fill = area / box_area
        circularity = (4.0 * math.pi * area / (perim * perim)) if perim > 0 else 0.0
        stats.append(
            {
                "box": (x, y, x + w, y + h),
                "area": area,
                "fill": fill,
                "circularity": circularity,
                "aspect": float(w) / float(h) if h > 0 else 0.0,
            }
        )
    return stats


def _ocr_token_boxes(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    if not OCR_ENABLED or pytesseract is None or Image is None:
        return []
    try:
        data = pytesseract.image_to_data(Image.fromarray(gray), output_type=pytesseract.Output.DICT)
    except Exception:
        return []
    boxes = []
    for i in range(len(data.get("text", []))):
        text = data["text"][i].strip()
        conf = float(data.get("conf", [0])[i]) if data.get("conf") else 0.0
        if not text or conf < 30:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if w * h < 100:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def _is_inside_any(box: Tuple[int, int, int, int], containers: List[Tuple[int, int, int, int]]) -> bool:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    for bx in containers:
        bx1, by1, bx2, by2 = bx
        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
            return True
    return False


def _box_mean(map_array: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    crop = map_array[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    return float(np.mean(crop))


def _box_mean_ring(map_array: np.ndarray, box: Tuple[int, int, int, int], pad: int) -> float:
    x1, y1, x2, y2 = box
    ox1 = max(0, x1 - pad)
    oy1 = max(0, y1 - pad)
    ox2 = min(map_array.shape[1], x2 + pad)
    oy2 = min(map_array.shape[0], y2 + pad)
    outer = map_array[oy1:oy2, ox1:ox2]
    if outer.size == 0:
        return 0.0
    inner = map_array[y1:y2, x1:x2]
    outer_sum = float(np.sum(outer))
    inner_sum = float(np.sum(inner))
    outer_count = float(outer.size)
    inner_count = float(inner.size)
    ring_count = outer_count - inner_count
    if ring_count <= 0:
        return 0.0
    return float((outer_sum - inner_sum) / ring_count)


def _local_texture_maps(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (31, 31))
    mean_sq = cv2.blur(gray_f * gray_f, (31, 31))
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    blur = cv2.GaussianBlur(gray_f, (0, 0), 3.0)
    residual = np.abs(gray_f - blur)
    grad_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    return residual, variance, gradient


def _block_energy(gray: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    crop = gray[y1:y2, x1:x2]
    if crop.size < 64 or cv2 is None:
        return 0.0
    crop_h = crop.shape[0] - (crop.shape[0] % 8)
    crop_w = crop.shape[1] - (crop.shape[1] % 8)
    if crop_h < 8 or crop_w < 8:
        return 0.0
    crop = crop[:crop_h, :crop_w].astype(np.float32)
    energies = []
    for yy in range(0, crop.shape[0], 8):
        for xx in range(0, crop.shape[1], 8):
            block = crop[yy:yy + 8, xx:xx + 8]
            block = block - float(np.mean(block))
            dct = cv2.dct(block)
            energies.append(float(np.mean(np.abs(dct[1:, 1:]))))
    if not energies:
        return 0.0
    return float(np.mean(energies))


def _jpeg_block_bonus(page: DocumentPage, gray: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    if page.image_path is None:
        return 0.0
    suffix = Path(page.image_path).suffix.lower()
    if suffix not in {".jpg", ".jpeg"}:
        return 0.0
    pad = max(12, min(24, max(box[2] - box[0], box[3] - box[1]) // 2))
    cand_energy = _block_energy(gray, box)
    x1, y1, x2, y2 = box
    ox1 = max(0, x1 - pad)
    oy1 = max(0, y1 - pad)
    ox2 = min(gray.shape[1], x2 + pad)
    oy2 = min(gray.shape[0], y2 + pad)
    outer = (ox1, oy1, ox2, oy2)
    outer_energy = _block_energy(gray, outer)
    if outer_energy <= 1e-6:
        return 0.0
    ratio = cand_energy / outer_energy
    if ratio < 0.7:
        return min(1.0, (0.7 - ratio) * 3.0)
    return 0.0


def _ocr_similarity(gray: np.ndarray, box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    if not OCR_ENABLED:
        return 1.0
    if pytesseract is None or Image is None:
        return 0.0

    def _crop_text(box: Tuple[int, int, int, int]) -> str:
        x1, y1, x2, y2 = box
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            return ""
        text = pytesseract.image_to_string(Image.fromarray(crop))
        return " ".join(text.split()).lower()

    text_a = _crop_text(box_a)
    text_b = _crop_text(box_b)
    if not text_a or not text_b:
        return 0.0
    return difflib.SequenceMatcher(None, text_a, text_b).ratio()


def _clip_box(box: Tuple[int, int, int, int], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _scale_box(box: Tuple[int, int, int, int], scale: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    if scale == 0:
        return box
    inv = 1.0 / scale
    return (
        int(round(x1 * inv)),
        int(round(y1 * inv)),
        int(round(x2 * inv)),
        int(round(y2 * inv)),
    )


def postprocess_regions(
    regions: List[DetectedRegion],
    predicted_categories: List[str],
    page: DocumentPage,
) -> List[DetectedRegion]:
    if not regions:
        return []
    width = page.image_width or 0
    height = page.image_height or 0
    cleaned: List[DetectedRegion] = []
    for region in regions:
        x1 = max(0, min(region.x, max(0, width - 1)))
        y1 = max(0, min(region.y, max(0, height - 1)))
        x2 = max(x1 + 1, min(region.x + region.w, width)) if width else region.x + region.w
        y2 = max(y1 + 1, min(region.y + region.h, height)) if height else region.y + region.h
        if (x2 - x1) * (y2 - y1) < 200:
            continue
        cleaned.append(
            DetectedRegion(
                x=int(x1),
                y=int(y1),
                w=int(x2 - x1),
                h=int(y2 - y1),
                category_id=region.category_id,
                type=region.type,
                stretch_factor=region.stretch_factor,
                header_source=region.header_source,
                body_source=region.body_source,
            )
        )
    return cleaned


def enrich_region_metadata(
    regions: List[DetectedRegion],
    predicted_categories: List[str],
    page: DocumentPage,
) -> List[DetectedRegion]:
    for r in regions:
        if r.category_id == "C3":
            r.type = "text"
        elif r.category_id == "C4":
            r.type = "removed_content"
        elif r.category_id == "C5":
            r.header_source = "suspected_merge"
        elif r.category_id == "C7":
            r.type = "irregular_spacing"
            r.stretch_factor = 1.25
        elif r.category_id == "C9":
            r.type = "edited_field"
    return regions


def validate_prediction(
    page: DocumentPage,
    predicted_categories: List[str],
    regions: List[DetectedRegion],
) -> Dict[str, Any]:
    valid = [c for c in predicted_categories if c in CATEGORY_IDS]
    requires_yaml = not set(valid).issubset(CATEGORY_ONLY_CLASSES)
    ok = True
    if requires_yaml and not regions:
        ok = False
    return {"ok": ok, "categories": valid, "requires_yaml": requires_yaml}


def build_page_analysis_result(
    page: DocumentPage,
    predicted_categories: List[str],
    regions: List[DetectedRegion],
    quality_summary: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
) -> PageAnalysisResult:
    return PageAnalysisResult(
        source_link=page.source_link,
        file_name=page.page_file_name,
        original_file_name=page.original_file_name,
        page_number=page.page_number,
        predicted_categories=predicted_categories,
        detected_regions=regions,
        notes={"quality": quality_summary or {}, "validation": validation or {}},
    )


# =========================
# C5 MERGE DETECTOR (ADD-ON)
# =========================

def _region_noise_profile(
    residual: np.ndarray,
    variance: np.ndarray,
    gradient: np.ndarray,
    box: Tuple[int, int, int, int],
    eps: float = 1e-6,
) -> np.ndarray:
    return np.array(
        [
            max(_box_mean(residual, box), eps),
            max(_box_mean(variance, box), eps),
            max(_box_mean(gradient, box), eps),
        ],
        dtype=np.float32,
    )


def _region_text_profile(
    mask: np.ndarray,
    edges: np.ndarray,
    token_boxes: List[Tuple[int, int, int, int]],
    box: Tuple[int, int, int, int],
    eps: float = 1e-6,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop_mask = mask[y1:y2, x1:x2]
    crop_edges = edges[y1:y2, x1:x2]
    text_density = float(np.mean(crop_mask > 0)) if crop_mask.size else 0.0
    edge_density = float(np.mean(crop_edges > 0)) if crop_edges.size else 0.0
    heights = []
    widths = []
    for bx in token_boxes:
        tx1, ty1, tx2, ty2 = bx
        cx = (tx1 + tx2) // 2
        cy = (ty1 + ty2) // 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            heights.append(max(1, ty2 - ty1))
            widths.append(max(1, tx2 - tx1))
    if heights:
        med_h = float(np.median(heights))
        med_w = float(np.median(widths))
    else:
        med_h = 0.0
        med_w = 0.0
    return np.array([max(text_density, eps), max(edge_density, eps), max(med_h, eps), max(med_w, eps)], dtype=np.float32)


def _profile_diff(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(a, b)
    return float(np.mean(np.abs(a - b) / denom))


def _profile_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(b, 1e-6)
    return float(np.mean(np.abs(a - b) / denom))


# =========================
# C6 WATERMARK REMOVAL (ADD-ON)
# =========================

def _band_periodicity(candidate: np.ndarray, angle_deg: float) -> float:
    h, w = candidate.shape[:2]
    if h < 30 or w < 30:
        return 0.0
    center = (w * 0.5, h * 0.5)
    rot = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    rotated = cv2.warpAffine(candidate, rot, (w, h), flags=cv2.INTER_NEAREST)
    row_density = (rotated > 0).mean(axis=1)
    if float(np.max(row_density)) < 0.01:
        return 0.0
    signal = row_density - float(np.mean(row_density))
    if np.allclose(signal, 0.0):
        return 0.0
    spectrum = np.abs(np.fft.rfft(signal))
    if spectrum.size < 4:
        return 0.0
    spectrum[0] = 0.0
    peak = float(np.max(spectrum))
    mean_val = float(np.mean(spectrum)) + 1e-6
    return peak / mean_val


# =========================
# C8/C9 AI-GENERATED DETECTOR (ADD-ON)
# =========================

def _fft_spectrum_features(gray: np.ndarray, max_dim: int = 512) -> Dict[str, float]:
    if gray is None:
        return {"flatness": 0.0, "peak_ratio": 0.0, "highfreq_ratio": 0.0}
    gray_small, _ = _resize_max(gray, max_dim=max_dim)
    if gray_small is None or gray_small.size < 1024:
        return {"flatness": 0.0, "peak_ratio": 0.0, "highfreq_ratio": 0.0}
    gray_f = gray_small.astype(np.float32)
    gray_f = gray_f - float(np.mean(gray_f))
    fft = np.fft.fft2(gray_f)
    power = np.abs(np.fft.fftshift(fft)) ** 2
    h, w = power.shape[:2]
    yy, xx = np.indices((h, w))
    cy = (h - 1) * 0.5
    cx = (w - 1) * 0.5
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = float(np.max(r)) + 1e-6
    r_norm = r / max_r
    band = r_norm > 0.15
    mid = (r_norm > 0.25) & (r_norm <= 0.6)
    high = (r_norm > 0.6) & (r_norm <= 0.95)
    if not np.any(high) or not np.any(mid):
        return {"flatness": 0.0, "peak_ratio": 0.0, "highfreq_ratio": 0.0}
    high_sum = float(np.sum(power[high]))
    mid_sum = float(np.sum(power[mid])) + 1e-6
    highfreq_ratio = high_sum / mid_sum
    high_power = power[high] + 1e-6
    peak_ratio = float(np.max(high_power) / float(np.mean(high_power)))
    band_power = power[band] + 1e-6
    flatness = float(np.exp(np.mean(np.log(band_power))) / np.mean(band_power))
    return {"flatness": flatness, "peak_ratio": peak_ratio, "highfreq_ratio": highfreq_ratio}


def _patch_fft_features(gray: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    crop = gray[y1:y2, x1:x2]
    if crop.size < 400:
        return 0.0, 0.0
    feats = _fft_spectrum_features(crop, max_dim=128)
    return feats["highfreq_ratio"], feats["peak_ratio"]


def _box_ring_ratio(
    map_array: np.ndarray,
    box: Tuple[int, int, int, int],
    pad: int,
    eps: float = 1e-6,
) -> float:
    inner = _box_mean(map_array, box)
    outer = _box_mean_ring(map_array, box, pad)
    return (inner + eps) / (outer + eps)


def _box_stroke_mean(mask: np.ndarray, dist: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    crop_mask = mask[y1:y2, x1:x2] > 0
    if crop_mask.size == 0 or not np.any(crop_mask):
        return 0.0
    crop_dist = dist[y1:y2, x1:x2]
    return float(np.mean(crop_dist[crop_mask]))


def _ai_box_profile(
    gray: np.ndarray,
    mask: np.ndarray,
    edges: np.ndarray,
    residual: np.ndarray,
    variance: np.ndarray,
    gradient: np.ndarray,
    dist: np.ndarray,
    box: Tuple[int, int, int, int],
) -> Dict[str, float]:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return {"edge": 0.0, "grad": 0.0, "res": 0.0, "var": 0.0, "stroke": 0.0, "fft_ratio": 0.0, "fft_peak": 0.0, "h": 0.0, "w": 0.0}
    crop_edges = edges[y1:y2, x1:x2]
    edge_density = float(np.mean(crop_edges > 0)) if crop_edges.size else 0.0
    grad_mean = _box_mean(gradient, box)
    res_mean = _box_mean(residual, box)
    var_mean = _box_mean(variance, box)
    stroke_mean = _box_stroke_mean(mask, dist, box)
    fft_ratio, fft_peak = _patch_fft_features(gray, box)
    return {
        "edge": edge_density,
        "grad": grad_mean,
        "res": res_mean,
        "var": var_mean,
        "stroke": stroke_mean,
        "fft_ratio": fft_ratio,
        "fft_peak": fft_peak,
        "h": float(y2 - y1),
        "w": float(x2 - x1),
    }


def _merge_line_boxes(
    boxes: List[Tuple[int, int, int, int]],
    pad_x: int,
    pad_y: int,
) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    merged: List[Tuple[int, int, int, int]] = []
    cur = list(boxes[0])
    for bx in boxes[1:]:
        if bx[0] <= cur[2] + pad_x:
            cur[0] = min(cur[0], bx[0])
            cur[1] = min(cur[1], bx[1])
            cur[2] = max(cur[2], bx[2])
            cur[3] = max(cur[3], bx[3])
        else:
            merged.append(tuple(cur))
            cur = list(bx)
    merged.append(tuple(cur))
    padded: List[Tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in merged:
        padded.append((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y))
    return padded


def _boxes_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _ai_region_coverage(
    regions: List[DetectedRegion],
    mask: np.ndarray,
    line_boxes: List[Tuple[int, int, int, int]],
) -> Tuple[float, float]:
    if not regions or mask is None:
        return 0.0, 0.0
    text_area = float(np.sum(mask > 0))
    region_area = float(sum(r.w * r.h for r in regions))
    coverage = region_area / max(1.0, text_area)
    if not line_boxes:
        return coverage, 0.0
    lines_hit = 0
    for line in line_boxes:
        for r in regions:
            rbox = (r.x, r.y, r.x + r.w, r.y + r.h)
            if _boxes_intersect(line, rbox):
                lines_hit += 1
                break
    line_cov = float(lines_hit) / float(max(len(line_boxes), 1))
    return coverage, line_cov


# =========================
# TUNING PRESETS (ADD-ON)
# =========================
DETECTOR_TUNING = { 'hyper_loose': { 'c1': { 'max_dim': 1200, 'orb_nfeatures': 5000, 'orb_scale_factor': 1.2, 'orb_nlevels': 8, 'match_ratio': 0.95, 'cluster_bin_size': 14.0, 'min_cluster_pairs': 4, 'min_keypoints': 8, 'min_area': 120, 'min_strength': 6, 'ocr_similarity': 0.35 }, 'c2': { 'canny_low': 40, 'canny_high': 140, 'edge_z': 1.2, 'stroke_z': 1.2, 'ocr_edge_density': 0.1, 'min_region_area': 40, 'min_line_components': 1, 'component_min_area': 40 }, 'c3': { 'component_min_area': 80, 'min_candidate_area': 100, 'stamp_circularity': 0.3, 'stamp_fill': 0.12, 'signature_aspect': 1.6, 'signature_fill_max': 0.5, 'fill_min': 0.2, 'ocr_min_area': 60, 'min_region_area': 60 }, 'c4': { 'gap_min_abs': 4.0, 'gap_median_mult': 0.9, 'gap_token_width_mult': 0.25, 'smooth_percentile': 65, 'smooth_min_area': 40, 'smooth_min_dim': 3, 'score_threshold': 0.6, 'min_region_area': 40, 'ring_var_ratio': 0.98, 'ring_grad_ratio': 0.95, 'ring_res_ratio': 0.98, 'ring_fg_delta': 0.005, 'erased_text_bonus': 0.9 }, 'c5': { 'row_density_thresh': 0.006, 'col_density_thresh': 0.003, 'gap_thresh_px': 6, 'gap_thresh_ratio': 0.03, 'band_min_height_px': 8, 'band_min_height_ratio': 0.03, 'band_min_width': 20, 'min_header_height_ratio': 0.04, 'cue_threshold': 0.12, 'cue_count_min': 1, 'score_threshold': 0.6, 'dist_balance_max': 0.35, 'dist_min': 0.3, 'min_region_area': 150 }, 'c6': { 'var_percentile': 55, 'grad_percentile': 55, 'res_percentile': 55, 'candidate_density_min': 0.0005, 'canny_low': 15, 'canny_high': 70, 'hough_threshold': 30, 'hough_min_len_ratio': 0.04, 'hough_min_len_px': 20, 'hough_max_gap': 28, 'diag_angle_min': 10.0, 'diag_angle_max': 80.0, 'diag_ratio_min': 0.15, 'diag_count_min': 2, 'periodicity_min': 1.3, 'contour_area_min': 50, 'aspect_min': 1.2, 'angle_delta_max': 60.0, 'box_area_min': 80, 'fallback_box_area_min': 250 }, 'c7': { 'min_tokens_page': 1, 'min_tokens_per_line': 1, 'min_gaps_per_line': 1, 'med_gap_min': 1.5, 'mad_gap_min': 0.5, 'gap_cv_max': 1.2, 'min_token_width': 2, 'large_gap_min_abs': 5.0, 'large_gap_median_mult': 1.2, 'large_gap_z': 1.5, 'large_gap_line_ratio_max': 0.45, 'tight_gap_median_mult': 0.7, 'tight_gap_median_min': 3.0, 'tight_gap_z': -1.2, 'single_gap_median_mult': 2.0, 'single_gap_min_abs': 7.0, 'single_gap_min_tokens': 3, 'pad_y_ratio': 0.25, 'pad_x_ratio': 0.1 }, 'c8': { 'text_density_min': 0.002, 'bg_var_ratio_max': 0.85, 'bg_res_ratio_max': 0.85, 'bg_grad_ratio_max': 0.95, 'stroke_cv_max': 0.4, 'height_cv_max': 0.3, 'spec_peak_min': 5.0, 'spec_highfreq_min': 1.2, 'spec_flatness_min': 0.35, 'bg_res_ratio_flatness_max': 0.92, 'coverage_min': 0.15, 'line_coverage_min': 0.25, 'coverage_high_min': 0.35, 'score_threshold': 1.4 }, 'c9': { 'canny_low': 35, 'canny_high': 100, 'min_token_area_line': 40, 'min_token_area_susp': 60, 'min_line_tokens': 1, 'z_edge': 1.4, 'z_grad': 1.4, 'z_res': 1.4, 'z_var': 1.4, 'z_stroke': 1.4, 'z_height': 1.6, 'ring_ratio_low': 0.95, 'ring_ratio_high': 1.15, 'fft_peak_min': 5.0, 'fft_ratio_min': 1.1, 'score_threshold': 1.4, 'min_region_area': 60, 'pad_x_ratio': 0.4, 'pad_y_ratio': 0.25, 'pad_x_min': 2, 'pad_y_min': 2 } }, 'current': { 'c1': { 'max_dim': 1200, 'orb_nfeatures': 4000, 'orb_scale_factor': 1.2, 'orb_nlevels': 8, 'match_ratio': 0.75, 'cluster_bin_size': 12.0, 'min_cluster_pairs': 12, 'min_keypoints': 20, 'min_area': 400, 'min_strength': 18, 'ocr_similarity': 0.65 }, 'c2': { 'canny_low': 50, 'canny_high': 150, 'edge_z': 2.5, 'stroke_z': 2.5, 'ocr_edge_density': 0.25, 'min_region_area': 150, 'min_line_components': 3, 'component_min_area': 120 }, 'c3': { 'component_min_area': 200, 'min_candidate_area': 300, 'stamp_circularity': 0.55, 'stamp_fill': 0.25, 'signature_aspect': 3.5, 'signature_fill_max': 0.25, 'fill_min': 0.4, 'ocr_min_area': 150, 'min_region_area': 150 }, 'c4': { 'gap_min_abs': 12.0, 'gap_median_mult': 1.6, 'gap_token_width_mult': 0.55, 'smooth_percentile': 35, 'smooth_min_area': 150, 'smooth_min_dim': 6, 'score_threshold': 2.0, 'min_region_area': 120, 'ring_var_ratio': 0.75, 'ring_grad_ratio': 0.72, 'ring_res_ratio': 0.8, 'ring_fg_delta': 0.05, 'erased_text_bonus': 0.6 }, 'c5': { 'row_density_thresh': 0.015, 'col_density_thresh': 0.01, 'gap_thresh_px': 12, 'gap_thresh_ratio': 0.06, 'band_min_height_px': 18, 'band_min_height_ratio': 0.06, 'band_min_width': 40, 'min_header_height_ratio': 0.08, 'cue_threshold': 0.25, 'cue_count_min': 2, 'score_threshold': 1.4, 'dist_balance_max': 0.15, 'dist_min': 0.6, 'min_region_area': 400 }, 'c6': { 'var_percentile': 30, 'grad_percentile': 30, 'res_percentile': 35, 'candidate_density_min': 0.003, 'canny_low': 40, 'canny_high': 120, 'hough_threshold': 60, 'hough_min_len_ratio': 0.08, 'hough_min_len_px': 40, 'hough_max_gap': 15, 'diag_angle_min': 20.0, 'diag_angle_max': 70.0, 'diag_ratio_min': 0.3, 'diag_count_min': 6, 'periodicity_min': 2.5, 'contour_area_min': 180, 'aspect_min': 2.0, 'angle_delta_max': 30.0, 'box_area_min': 200, 'fallback_box_area_min': 600 }, 'c7': { 'min_tokens_page': 4, 'min_tokens_per_line': 4, 'min_gaps_per_line': 3, 'med_gap_min': 4.0, 'mad_gap_min': 1.0, 'gap_cv_max': 0.8, 'min_token_width': 6, 'large_gap_min_abs': 12.0, 'large_gap_median_mult': 2.0, 'large_gap_z': 3.0, 'large_gap_line_ratio_max': 0.28, 'tight_gap_median_mult': 0.45, 'tight_gap_median_min': 8.0, 'tight_gap_z': -2.5, 'single_gap_median_mult': 3.5, 'single_gap_min_abs': 18.0, 'single_gap_min_tokens': 6, 'pad_y_ratio': 0.25, 'pad_x_ratio': 0.1 }, 'c8': { 'text_density_min': 0.008, 'bg_var_ratio_max': 0.6, 'bg_res_ratio_max': 0.6, 'bg_grad_ratio_max': 0.7, 'stroke_cv_max': 0.22, 'height_cv_max': 0.18, 'spec_peak_min': 10.0, 'spec_highfreq_min': 2.2, 'spec_flatness_min': 0.48, 'bg_res_ratio_flatness_max': 0.75, 'coverage_min': 0.35, 'line_coverage_min': 0.5, 'coverage_high_min': 0.6, 'score_threshold': 2.8 }, 'c9': { 'canny_low': 60, 'canny_high': 160, 'min_token_area_line': 90, 'min_token_area_susp': 120, 'min_line_tokens': 3, 'z_edge': 2.6, 'z_grad': 2.6, 'z_res': 2.6, 'z_var': 2.6, 'z_stroke': 2.6, 'z_height': 2.8, 'ring_ratio_low': 0.6, 'ring_ratio_high': 1.6, 'fft_peak_min': 10.0, 'fft_ratio_min': 2.0, 'score_threshold': 2.6, 'min_region_area': 160, 'pad_x_ratio': 0.4, 'pad_y_ratio': 0.25, 'pad_x_min': 3, 'pad_y_min': 2 } }, 'normal': { 'c1': { 'max_dim': 1200, 'orb_nfeatures': 4000, 'orb_scale_factor': 1.2, 'orb_nlevels': 8, 'match_ratio': 0.73, 'cluster_bin_size': 12.0, 'min_cluster_pairs': 14, 'min_keypoints': 24, 'min_area': 450, 'min_strength': 20, 'ocr_similarity': 0.7 }, 'c2': { 'canny_low': 50, 'canny_high': 150, 'edge_z': 2.7, 'stroke_z': 2.7, 'ocr_edge_density': 0.27, 'min_region_area': 170, 'min_line_components': 3, 'component_min_area': 130 }, 'c3': { 'component_min_area': 220, 'min_candidate_area': 340, 'stamp_circularity': 0.58, 'stamp_fill': 0.28, 'signature_aspect': 3.8, 'signature_fill_max': 0.23, 'fill_min': 0.43, 'ocr_min_area': 170, 'min_region_area': 170 }, 'c4': { 'gap_min_abs': 12.0, 'gap_median_mult': 1.75, 'gap_token_width_mult': 0.6, 'smooth_percentile': 33, 'smooth_min_area': 170, 'smooth_min_dim': 7, 'score_threshold': 2.2, 'min_region_area': 140, 'ring_var_ratio': 0.72, 'ring_grad_ratio': 0.7, 'ring_res_ratio': 0.77, 'ring_fg_delta': 0.06, 'erased_text_bonus': 0.55 }, 'c5': { 'row_density_thresh': 0.016, 'col_density_thresh': 0.012, 'gap_thresh_px': 12, 'gap_thresh_ratio': 0.065, 'band_min_height_px': 18, 'band_min_height_ratio': 0.065, 'band_min_width': 40, 'min_header_height_ratio': 0.085, 'cue_threshold': 0.27, 'cue_count_min': 2, 'score_threshold': 1.5, 'dist_balance_max': 0.14, 'dist_min': 0.62, 'min_region_area': 440 }, 'c6': { 'var_percentile': 28, 'grad_percentile': 28, 'res_percentile': 32, 'candidate_density_min': 0.0035, 'canny_low': 40, 'canny_high': 120, 'hough_threshold': 62, 'hough_min_len_ratio': 0.085, 'hough_min_len_px': 40, 'hough_max_gap': 14, 'diag_angle_min': 22.0, 'diag_angle_max': 68.0, 'diag_ratio_min': 0.32, 'diag_count_min': 7, 'periodicity_min': 2.7, 'contour_area_min': 200, 'aspect_min': 2.2, 'angle_delta_max': 28.0, 'box_area_min': 240, 'fallback_box_area_min': 700 }, 'c7': { 'min_tokens_page': 4, 'min_tokens_per_line': 4, 'min_gaps_per_line': 3, 'med_gap_min': 4.5, 'mad_gap_min': 1.1, 'gap_cv_max': 0.75, 'min_token_width': 6, 'large_gap_min_abs': 13.0, 'large_gap_median_mult': 2.2, 'large_gap_z': 3.2, 'large_gap_line_ratio_max': 0.26, 'tight_gap_median_mult': 0.4, 'tight_gap_median_min': 9.0, 'tight_gap_z': -2.7, 'single_gap_median_mult': 3.7, 'single_gap_min_abs': 20.0, 'single_gap_min_tokens': 6, 'pad_y_ratio': 0.25, 'pad_x_ratio': 0.1 }, 'c8': { 'text_density_min': 0.009, 'bg_var_ratio_max': 0.58, 'bg_res_ratio_max': 0.58, 'bg_grad_ratio_max': 0.68, 'stroke_cv_max': 0.2, 'height_cv_max': 0.17, 'spec_peak_min': 11.0, 'spec_highfreq_min': 2.4, 'spec_flatness_min': 0.5, 'bg_res_ratio_flatness_max': 0.72, 'coverage_min': 0.38, 'line_coverage_min': 0.55, 'coverage_high_min': 0.65, 'score_threshold': 3.0 }, 'c9': { 'canny_low': 60, 'canny_high': 160, 'min_token_area_line': 100, 'min_token_area_susp': 140, 'min_line_tokens': 3, 'z_edge': 2.8, 'z_grad': 2.8, 'z_res': 2.8, 'z_var': 2.8, 'z_stroke': 2.8, 'z_height': 3.0, 'ring_ratio_low': 0.58, 'ring_ratio_high': 1.62, 'fft_peak_min': 11.0, 'fft_ratio_min': 2.2, 'score_threshold': 2.8, 'min_region_area': 180, 'pad_x_ratio': 0.4, 'pad_y_ratio': 0.25, 'pad_x_min': 3, 'pad_y_min': 2 } }, 'strict': { 'c1': { 'max_dim': 1200, 'orb_nfeatures': 4000, 'orb_scale_factor': 1.2, 'orb_nlevels': 8, 'match_ratio': 0.7, 'cluster_bin_size': 12.0, 'min_cluster_pairs': 16, 'min_keypoints': 28, 'min_area': 500, 'min_strength': 22, 'ocr_similarity': 0.75 }, 'c2': { 'canny_low': 50, 'canny_high': 150, 'edge_z': 3.0, 'stroke_z': 3.0, 'ocr_edge_density': 0.3, 'min_region_area': 200, 'min_line_components': 4, 'component_min_area': 140 }, 'c3': { 'component_min_area': 250, 'min_candidate_area': 380, 'stamp_circularity': 0.62, 'stamp_fill': 0.3, 'signature_aspect': 4.2, 'signature_fill_max': 0.22, 'fill_min': 0.46, 'ocr_min_area': 190, 'min_region_area': 190 }, 'c4': { 'gap_min_abs': 14.0, 'gap_median_mult': 2.0, 'gap_token_width_mult': 0.7, 'smooth_percentile': 30, 'smooth_min_area': 220, 'smooth_min_dim': 8, 'score_threshold': 2.6, 'min_region_area': 180, 'ring_var_ratio': 0.68, 'ring_grad_ratio': 0.66, 'ring_res_ratio': 0.72, 'ring_fg_delta': 0.08, 'erased_text_bonus': 0.5 }, 'c5': { 'row_density_thresh': 0.018, 'col_density_thresh': 0.015, 'gap_thresh_px': 12, 'gap_thresh_ratio': 0.07, 'band_min_height_px': 18, 'band_min_height_ratio': 0.07, 'band_min_width': 40, 'min_header_height_ratio': 0.095, 'cue_threshold': 0.3, 'cue_count_min': 2, 'score_threshold': 1.7, 'dist_balance_max': 0.12, 'dist_min': 0.7, 'min_region_area': 520 }, 'c6': { 'var_percentile': 25, 'grad_percentile': 25, 'res_percentile': 30, 'candidate_density_min': 0.0045, 'canny_low': 45, 'canny_high': 130, 'hough_threshold': 65, 'hough_min_len_ratio': 0.09, 'hough_min_len_px': 40, 'hough_max_gap': 12, 'diag_angle_min': 24.0, 'diag_angle_max': 66.0, 'diag_ratio_min': 0.35, 'diag_count_min': 8, 'periodicity_min': 3.0, 'contour_area_min': 240, 'aspect_min': 2.4, 'angle_delta_max': 25.0, 'box_area_min': 280, 'fallback_box_area_min': 800 }, 'c7': { 'min_tokens_page': 4, 'min_tokens_per_line': 5, 'min_gaps_per_line': 3, 'med_gap_min': 5.0, 'mad_gap_min': 1.2, 'gap_cv_max': 0.7, 'min_token_width': 6, 'large_gap_min_abs': 14.0, 'large_gap_median_mult': 2.4, 'large_gap_z': 3.5, 'large_gap_line_ratio_max': 0.24, 'tight_gap_median_mult': 0.38, 'tight_gap_median_min': 10.0, 'tight_gap_z': -3.0, 'single_gap_median_mult': 4.0, 'single_gap_min_abs': 22.0, 'single_gap_min_tokens': 6, 'pad_y_ratio': 0.25, 'pad_x_ratio': 0.1 }, 'c8': { 'text_density_min': 0.01, 'bg_var_ratio_max': 0.55, 'bg_res_ratio_max': 0.55, 'bg_grad_ratio_max': 0.65, 'stroke_cv_max': 0.18, 'height_cv_max': 0.16, 'spec_peak_min': 12.0, 'spec_highfreq_min': 2.6, 'spec_flatness_min': 0.52, 'bg_res_ratio_flatness_max': 0.7, 'coverage_min': 0.42, 'line_coverage_min': 0.6, 'coverage_high_min': 0.7, 'score_threshold': 3.3 }, 'c9': { 'canny_low': 60, 'canny_high': 160, 'min_token_area_line': 110, 'min_token_area_susp': 160, 'min_line_tokens': 4, 'z_edge': 3.0, 'z_grad': 3.0, 'z_res': 3.0, 'z_var': 3.0, 'z_stroke': 3.0, 'z_height': 3.2, 'ring_ratio_low': 0.55, 'ring_ratio_high': 1.65, 'fft_peak_min': 12.0, 'fft_ratio_min': 2.4, 'score_threshold': 3.1, 'min_region_area': 200, 'pad_x_ratio': 0.4, 'pad_y_ratio': 0.25, 'pad_x_min': 3, 'pad_y_min': 2 } }, 'very_strict': { 'c1': { 'max_dim': 1200, 'orb_nfeatures': 4000, 'orb_scale_factor': 1.2, 'orb_nlevels': 8, 'match_ratio': 0.68, 'cluster_bin_size': 12.0, 'min_cluster_pairs': 20, 'min_keypoints': 32, 'min_area': 600, 'min_strength': 25, 'ocr_similarity': 0.8 }, 'c2': { 'canny_low': 50, 'canny_high': 150, 'edge_z': 3.5, 'stroke_z': 3.5, 'ocr_edge_density': 0.35, 'min_region_area': 240, 'min_line_components': 4, 'component_min_area': 160 }, 'c3': { 'component_min_area': 280, 'min_candidate_area': 420, 'stamp_circularity': 0.66, 'stamp_fill': 0.33, 'signature_aspect': 4.6, 'signature_fill_max': 0.2, 'fill_min': 0.5, 'ocr_min_area': 220, 'min_region_area': 220 }, 'c4': { 'gap_min_abs': 16.0, 'gap_median_mult': 2.4, 'gap_token_width_mult': 0.75, 'smooth_percentile': 25, 'smooth_min_area': 280, 'smooth_min_dim': 10, 'score_threshold': 3.0, 'min_region_area': 220, 'ring_var_ratio': 0.62, 'ring_grad_ratio': 0.6, 'ring_res_ratio': 0.65, 'ring_fg_delta': 0.1, 'erased_text_bonus': 0.4 }, 'c5': { 'row_density_thresh': 0.02, 'col_density_thresh': 0.018, 'gap_thresh_px': 12, 'gap_thresh_ratio': 0.075, 'band_min_height_px': 18, 'band_min_height_ratio': 0.075, 'band_min_width': 40, 'min_header_height_ratio': 0.11, 'cue_threshold': 0.34, 'cue_count_min': 2, 'score_threshold': 2.0, 'dist_balance_max': 0.1, 'dist_min': 0.75, 'min_region_area': 600 }, 'c6': { 'var_percentile': 22, 'grad_percentile': 22, 'res_percentile': 27, 'candidate_density_min': 0.006, 'canny_low': 50, 'canny_high': 140, 'hough_threshold': 70, 'hough_min_len_ratio': 0.095, 'hough_min_len_px': 40, 'hough_max_gap': 10, 'diag_angle_min': 26.0, 'diag_angle_max': 64.0, 'diag_ratio_min': 0.38, 'diag_count_min': 10, 'periodicity_min': 3.4, 'contour_area_min': 280, 'aspect_min': 2.8, 'angle_delta_max': 22.0, 'box_area_min': 340, 'fallback_box_area_min': 900 }, 'c7': { 'min_tokens_page': 4, 'min_tokens_per_line': 5, 'min_gaps_per_line': 3, 'med_gap_min': 6.0, 'mad_gap_min': 1.4, 'gap_cv_max': 0.65, 'min_token_width': 6, 'large_gap_min_abs': 16.0, 'large_gap_median_mult': 2.6, 'large_gap_z': 3.8, 'large_gap_line_ratio_max': 0.22, 'tight_gap_median_mult': 0.35, 'tight_gap_median_min': 11.0, 'tight_gap_z': -3.2, 'single_gap_median_mult': 4.4, 'single_gap_min_abs': 24.0, 'single_gap_min_tokens': 6, 'pad_y_ratio': 0.25, 'pad_x_ratio': 0.1 }, 'c8': { 'text_density_min': 0.012, 'bg_var_ratio_max': 0.52, 'bg_res_ratio_max': 0.52, 'bg_grad_ratio_max': 0.62, 'stroke_cv_max': 0.16, 'height_cv_max': 0.15, 'spec_peak_min': 13.5, 'spec_highfreq_min': 2.8, 'spec_flatness_min': 0.54, 'bg_res_ratio_flatness_max': 0.68, 'coverage_min': 0.48, 'line_coverage_min': 0.65, 'coverage_high_min': 0.75, 'score_threshold': 3.6 }, 'c9': { 'canny_low': 60, 'canny_high': 160, 'min_token_area_line': 130, 'min_token_area_susp': 180, 'min_line_tokens': 4, 'z_edge': 3.3, 'z_grad': 3.3, 'z_res': 3.3, 'z_var': 3.3, 'z_stroke': 3.3, 'z_height': 3.4, 'ring_ratio_low': 0.52, 'ring_ratio_high': 1.7, 'fft_peak_min': 13.0, 'fft_ratio_min': 2.7, 'score_threshold': 3.4, 'min_region_area': 220, 'pad_x_ratio': 0.4, 'pad_y_ratio': 0.25, 'pad_x_min': 3, 'pad_y_min': 2 } }, 'loose': { 'c1': { 'max_dim': 1200, 'orb_nfeatures': 4000, 'orb_scale_factor': 1.2, 'orb_nlevels': 8, 'match_ratio': 0.8, 'cluster_bin_size': 12.0, 'min_cluster_pairs': 9, 'min_keypoints': 16, 'min_area': 300, 'min_strength': 14, 'ocr_similarity': 0.55 }, 'c2': { 'canny_low': 50, 'canny_high': 150, 'edge_z': 2.0, 'stroke_z': 2.0, 'ocr_edge_density': 0.2, 'min_region_area': 120, 'min_line_components': 2, 'component_min_area': 100 }, 'c3': { 'component_min_area': 160, 'min_candidate_area': 240, 'stamp_circularity': 0.5, 'stamp_fill': 0.22, 'signature_aspect': 3.0, 'signature_fill_max': 0.3, 'fill_min': 0.35, 'ocr_min_area': 120, 'min_region_area': 120 }, 'c4': { 'gap_min_abs': 10.0, 'gap_median_mult': 1.3, 'gap_token_width_mult': 0.45, 'smooth_percentile': 45, 'smooth_min_area': 120, 'smooth_min_dim': 5, 'score_threshold': 1.6, 'min_region_area': 100, 'ring_var_ratio': 0.82, 'ring_grad_ratio': 0.8, 'ring_res_ratio': 0.85, 'ring_fg_delta': 0.03, 'erased_text_bonus': 0.7 }, 'c5': { 'row_density_thresh': 0.012, 'col_density_thresh': 0.008, 'gap_thresh_px': 10, 'gap_thresh_ratio': 0.05, 'band_min_height_px': 16, 'band_min_height_ratio': 0.05, 'band_min_width': 35, 'min_header_height_ratio': 0.07, 'cue_threshold': 0.22, 'cue_count_min': 2, 'score_threshold': 1.2, 'dist_balance_max': 0.2, 'dist_min': 0.5, 'min_region_area': 320 }, 'c6': { 'var_percentile': 35, 'grad_percentile': 35, 'res_percentile': 40, 'candidate_density_min': 0.002, 'canny_low': 35, 'canny_high': 110, 'hough_threshold': 55, 'hough_min_len_ratio': 0.07, 'hough_min_len_px': 35, 'hough_max_gap': 18, 'diag_angle_min': 18.0, 'diag_angle_max': 72.0, 'diag_ratio_min': 0.25, 'diag_count_min': 5, 'periodicity_min': 2.1, 'contour_area_min': 140, 'aspect_min': 1.8, 'angle_delta_max': 35.0, 'box_area_min': 160, 'fallback_box_area_min': 500 }, 'c7': { 'min_tokens_page': 3, 'min_tokens_per_line': 3, 'min_gaps_per_line': 2, 'med_gap_min': 3.5, 'mad_gap_min': 0.9, 'gap_cv_max': 0.85, 'min_token_width': 5, 'large_gap_min_abs': 10.0, 'large_gap_median_mult': 1.7, 'large_gap_z': 2.6, 'large_gap_line_ratio_max': 0.32, 'tight_gap_median_mult': 0.5, 'tight_gap_median_min': 6.0, 'tight_gap_z': -2.1, 'single_gap_median_mult': 3.0, 'single_gap_min_abs': 14.0, 'single_gap_min_tokens': 5, 'pad_y_ratio': 0.25, 'pad_x_ratio': 0.1 }, 'c8': { 'text_density_min': 0.006, 'bg_var_ratio_max': 0.65, 'bg_res_ratio_max': 0.65, 'bg_grad_ratio_max': 0.75, 'stroke_cv_max': 0.26, 'height_cv_max': 0.2, 'spec_peak_min': 9.0, 'spec_highfreq_min': 2.0, 'spec_flatness_min': 0.46, 'bg_res_ratio_flatness_max': 0.8, 'coverage_min': 0.3, 'line_coverage_min': 0.45, 'coverage_high_min': 0.55, 'score_threshold': 2.4 }, 'c9': { 'canny_low': 55, 'canny_high': 150, 'min_token_area_line': 80, 'min_token_area_susp': 100, 'min_line_tokens': 2, 'z_edge': 2.2, 'z_grad': 2.2, 'z_res': 2.2, 'z_var': 2.2, 'z_stroke': 2.2, 'z_height': 2.5, 'ring_ratio_low': 0.7, 'ring_ratio_high': 1.4, 'fft_peak_min': 9.0, 'fft_ratio_min': 1.7, 'score_threshold': 2.2, 'min_region_area': 140, 'pad_x_ratio': 0.4, 'pad_y_ratio': 0.25, 'pad_x_min': 3, 'pad_y_min': 2 } }, 'very_loose': { 'c1': { 'max_dim': 1200, 'orb_nfeatures': 4000, 'orb_scale_factor': 1.2, 'orb_nlevels': 8, 'match_ratio': 0.85, 'cluster_bin_size': 12.0, 'min_cluster_pairs': 8, 'min_keypoints': 14, 'min_area': 250, 'min_strength': 12, 'ocr_similarity': 0.5 }, 'c2': { 'canny_low': 50, 'canny_high': 150, 'edge_z': 1.8, 'stroke_z': 1.8, 'ocr_edge_density': 0.18, 'min_region_area': 100, 'min_line_components': 2, 'component_min_area': 90 }, 'c3': { 'component_min_area': 140, 'min_candidate_area': 210, 'stamp_circularity': 0.45, 'stamp_fill': 0.2, 'signature_aspect': 2.6, 'signature_fill_max': 0.35, 'fill_min': 0.3, 'ocr_min_area': 100, 'min_region_area': 100 }, 'c4': { 'gap_min_abs': 8.0, 'gap_median_mult': 1.15, 'gap_token_width_mult': 0.4, 'smooth_percentile': 50, 'smooth_min_area': 90, 'smooth_min_dim': 4, 'score_threshold': 1.3, 'min_region_area': 80, 'ring_var_ratio': 0.88, 'ring_grad_ratio': 0.86, 'ring_res_ratio': 0.9, 'ring_fg_delta': 0.02, 'erased_text_bonus': 0.75 }, 'c5': { 'row_density_thresh': 0.01, 'col_density_thresh': 0.006, 'gap_thresh_px': 9, 'gap_thresh_ratio': 0.045, 'band_min_height_px': 14, 'band_min_height_ratio': 0.045, 'band_min_width': 32, 'min_header_height_ratio': 0.06, 'cue_threshold': 0.2, 'cue_count_min': 2, 'score_threshold': 1.05, 'dist_balance_max': 0.24, 'dist_min': 0.45, 'min_region_area': 260 }, 'c6': { 'var_percentile': 40, 'grad_percentile': 40, 'res_percentile': 45, 'candidate_density_min': 0.0015, 'canny_low': 30, 'canny_high': 100, 'hough_threshold': 50, 'hough_min_len_ratio': 0.065, 'hough_min_len_px': 30, 'hough_max_gap': 20, 'diag_angle_min': 16.0, 'diag_angle_max': 74.0, 'diag_ratio_min': 0.22, 'diag_count_min': 4, 'periodicity_min': 1.9, 'contour_area_min': 110, 'aspect_min': 1.6, 'angle_delta_max': 40.0, 'box_area_min': 140, 'fallback_box_area_min': 420 }, 'c7': { 'min_tokens_page': 3, 'min_tokens_per_line': 3, 'min_gaps_per_line': 2, 'med_gap_min': 3.0, 'mad_gap_min': 0.8, 'gap_cv_max': 0.9, 'min_token_width': 4, 'large_gap_min_abs': 9.0, 'large_gap_median_mult': 1.5, 'large_gap_z': 2.2, 'large_gap_line_ratio_max': 0.35, 'tight_gap_median_mult': 0.55, 'tight_gap_median_min': 5.0, 'tight_gap_z': -1.8, 'single_gap_median_mult': 2.7, 'single_gap_min_abs': 12.0, 'single_gap_min_tokens': 4, 'pad_y_ratio': 0.25, 'pad_x_ratio': 0.1 }, 'c8': { 'text_density_min': 0.005, 'bg_var_ratio_max': 0.7, 'bg_res_ratio_max': 0.7, 'bg_grad_ratio_max': 0.8, 'stroke_cv_max': 0.3, 'height_cv_max': 0.23, 'spec_peak_min': 8.0, 'spec_highfreq_min': 1.8, 'spec_flatness_min': 0.44, 'bg_res_ratio_flatness_max': 0.85, 'coverage_min': 0.25, 'line_coverage_min': 0.4, 'coverage_high_min': 0.5, 'score_threshold': 2.1 }, 'c9': { 'canny_low': 50, 'canny_high': 140, 'min_token_area_line': 70, 'min_token_area_susp': 90, 'min_line_tokens': 2, 'z_edge': 2.0, 'z_grad': 2.0, 'z_res': 2.0, 'z_var': 2.0, 'z_stroke': 2.0, 'z_height': 2.2, 'ring_ratio_low': 0.75, 'ring_ratio_high': 1.3, 'fft_peak_min': 8.0, 'fft_ratio_min': 1.5, 'score_threshold': 2.0, 'min_region_area': 120, 'pad_x_ratio': 0.4, 'pad_y_ratio': 0.25, 'pad_x_min': 3, 'pad_y_min': 2 } }, 'ultra_loose': { 'c1': { 'max_dim': 1200, 'orb_nfeatures': 4000, 'orb_scale_factor': 1.2, 'orb_nlevels': 8, 'match_ratio': 0.9, 'cluster_bin_size': 12.0, 'min_cluster_pairs': 6, 'min_keypoints': 12, 'min_area': 200, 'min_strength': 10, 'ocr_similarity': 0.45 }, 'c2': { 'canny_low': 50, 'canny_high': 150, 'edge_z': 1.6, 'stroke_z': 1.6, 'ocr_edge_density': 0.16, 'min_region_area': 80, 'min_line_components': 2, 'component_min_area': 80 }, 'c3': { 'component_min_area': 120, 'min_candidate_area': 180, 'stamp_circularity': 0.4, 'stamp_fill': 0.18, 'signature_aspect': 2.2, 'signature_fill_max': 0.38, 'fill_min': 0.28, 'ocr_min_area': 90, 'min_region_area': 90 }, 'c4': { 'gap_min_abs': 7.0, 'gap_median_mult': 1.1, 'gap_token_width_mult': 0.35, 'smooth_percentile': 55, 'smooth_min_area': 70, 'smooth_min_dim': 4, 'score_threshold': 1.1, 'min_region_area': 70, 'ring_var_ratio': 0.92, 'ring_grad_ratio': 0.9, 'ring_res_ratio': 0.93, 'ring_fg_delta': 0.015, 'erased_text_bonus': 0.8 }, 'c5': { 'row_density_thresh': 0.009, 'col_density_thresh': 0.005, 'gap_thresh_px': 8, 'gap_thresh_ratio': 0.04, 'band_min_height_px': 12, 'band_min_height_ratio': 0.04, 'band_min_width': 28, 'min_header_height_ratio': 0.055, 'cue_threshold': 0.18, 'cue_count_min': 2, 'score_threshold': 0.9, 'dist_balance_max': 0.28, 'dist_min': 0.4, 'min_region_area': 220 }, 'c6': { 'var_percentile': 45, 'grad_percentile': 45, 'res_percentile': 50, 'candidate_density_min': 0.001, 'canny_low': 25, 'canny_high': 90, 'hough_threshold': 45, 'hough_min_len_ratio': 0.06, 'hough_min_len_px': 28, 'hough_max_gap': 22, 'diag_angle_min': 15.0, 'diag_angle_max': 75.0, 'diag_ratio_min': 0.2, 'diag_count_min': 4, 'periodicity_min': 1.7, 'contour_area_min': 90, 'aspect_min': 1.5, 'angle_delta_max': 45.0, 'box_area_min': 120, 'fallback_box_area_min': 380 }, 'c7': { 'min_tokens_page': 2, 'min_tokens_per_line': 2, 'min_gaps_per_line': 2, 'med_gap_min': 2.5, 'mad_gap_min': 0.7, 'gap_cv_max': 0.95, 'min_token_width': 4, 'large_gap_min_abs': 8.0, 'large_gap_median_mult': 1.4, 'large_gap_z': 2.0, 'large_gap_line_ratio_max': 0.38, 'tight_gap_median_mult': 0.6, 'tight_gap_median_min': 4.5, 'tight_gap_z': -1.6, 'single_gap_median_mult': 2.4, 'single_gap_min_abs': 10.0, 'single_gap_min_tokens': 4, 'pad_y_ratio': 0.25, 'pad_x_ratio': 0.1 }, 'c8': { 'text_density_min': 0.004, 'bg_var_ratio_max': 0.75, 'bg_res_ratio_max': 0.75, 'bg_grad_ratio_max': 0.85, 'stroke_cv_max': 0.34, 'height_cv_max': 0.25, 'spec_peak_min': 7.0, 'spec_highfreq_min': 1.6, 'spec_flatness_min': 0.42, 'bg_res_ratio_flatness_max': 0.88, 'coverage_min': 0.22, 'line_coverage_min': 0.35, 'coverage_high_min': 0.45, 'score_threshold': 1.9 }, 'c9': { 'canny_low': 45, 'canny_high': 130, 'min_token_area_line': 60, 'min_token_area_susp': 80, 'min_line_tokens': 2, 'z_edge': 1.8, 'z_grad': 1.8, 'z_res': 1.8, 'z_var': 1.8, 'z_stroke': 1.8, 'z_height': 2.0, 'ring_ratio_low': 0.85, 'ring_ratio_high': 1.25, 'fft_peak_min': 7.0, 'fft_ratio_min': 1.3, 'score_threshold': 1.8, 'min_region_area': 100, 'pad_x_ratio': 0.4, 'pad_y_ratio': 0.25, 'pad_x_min': 3, 'pad_y_min': 2 } }, 'selective_loose_v2': { 'c1': { 'max_dim': 1200, 'orb_nfeatures': 4000, 'orb_scale_factor': 1.2, 'orb_nlevels': 8, 'match_ratio': 0.9, 'cluster_bin_size': 12.0, 'min_cluster_pairs': 6, 'min_keypoints': 12, 'min_area': 200, 'min_strength': 10, 'ocr_similarity': 0.45 }, 'c2': { 'canny_low': 50, 'canny_high': 150, 'edge_z': 2.4, 'stroke_z': 2.4, 'ocr_edge_density': 0.22, 'min_region_area': 130, 'min_line_components': 3, 'component_min_area': 110 }, 'c3': { 'component_min_area': 120, 'min_candidate_area': 180, 'stamp_circularity': 0.4, 'stamp_fill': 0.18, 'signature_aspect': 2.2, 'signature_fill_max': 0.38, 'fill_min': 0.28, 'ocr_min_area': 90, 'min_region_area': 90 }, 'c4': { 'gap_min_abs': 10.0, 'gap_median_mult': 1.35, 'gap_token_width_mult': 0.45, 'smooth_percentile': 40, 'smooth_min_area': 110, 'smooth_min_dim': 5, 'score_threshold': 1.6, 'min_region_area': 110, 'ring_var_ratio': 0.8, 'ring_grad_ratio': 0.78, 'ring_res_ratio': 0.85, 'ring_fg_delta': 0.03, 'erased_text_bonus': 0.7 }, 'c5': { 'row_density_thresh': 0.009, 'col_density_thresh': 0.005, 'gap_thresh_px': 8, 'gap_thresh_ratio': 0.04, 'band_min_height_px': 12, 'band_min_height_ratio': 0.04, 'band_min_width': 28, 'min_header_height_ratio': 0.055, 'cue_threshold': 0.18, 'cue_count_min': 2, 'score_threshold': 0.9, 'dist_balance_max': 0.28, 'dist_min': 0.4, 'min_region_area': 220 }, 'c6': { 'var_percentile': 45, 'grad_percentile': 45, 'res_percentile': 50, 'candidate_density_min': 0.001, 'canny_low': 25, 'canny_high': 90, 'hough_threshold': 45, 'hough_min_len_ratio': 0.06, 'hough_min_len_px': 28, 'hough_max_gap': 22, 'diag_angle_min': 15.0, 'diag_angle_max': 75.0, 'diag_ratio_min': 0.2, 'diag_count_min': 4, 'periodicity_min': 1.7, 'contour_area_min': 90, 'aspect_min': 1.5, 'angle_delta_max': 45.0, 'box_area_min': 120, 'fallback_box_area_min': 380 }, 'c7': { 'min_tokens_page': 2, 'min_tokens_per_line': 2, 'min_gaps_per_line': 2, 'med_gap_min': 2.5, 'mad_gap_min': 0.7, 'gap_cv_max': 0.95, 'min_token_width': 4, 'large_gap_min_abs': 8.0, 'large_gap_median_mult': 1.4, 'large_gap_z': 2.0, 'large_gap_line_ratio_max': 0.38, 'tight_gap_median_mult': 0.6, 'tight_gap_median_min': 4.5, 'tight_gap_z': -1.6, 'single_gap_median_mult': 2.4, 'single_gap_min_abs': 10.0, 'single_gap_min_tokens': 4, 'pad_y_ratio': 0.25, 'pad_x_ratio': 0.1 }, 'c8': { 'text_density_min': 0.004, 'bg_var_ratio_max': 0.75, 'bg_res_ratio_max': 0.75, 'bg_grad_ratio_max': 0.85, 'stroke_cv_max': 0.34, 'height_cv_max': 0.25, 'spec_peak_min': 7.0, 'spec_highfreq_min': 1.6, 'spec_flatness_min': 0.42, 'bg_res_ratio_flatness_max': 0.88, 'coverage_min': 0.22, 'line_coverage_min': 0.35, 'coverage_high_min': 0.45, 'score_threshold': 1.9 }, 'c9': { 'canny_low': 50, 'canny_high': 140, 'min_token_area_line': 80, 'min_token_area_susp': 100, 'min_line_tokens': 3, 'z_edge': 2.3, 'z_grad': 2.3, 'z_res': 2.3, 'z_var': 2.3, 'z_stroke': 2.3, 'z_height': 2.5, 'ring_ratio_low': 0.75, 'ring_ratio_high': 1.35, 'fft_peak_min': 8.5, 'fft_ratio_min': 1.7, 'score_threshold': 2.3, 'min_region_area': 140, 'pad_x_ratio': 0.4, 'pad_y_ratio': 0.25, 'pad_x_min': 3, 'pad_y_min': 2 } }, 'super_loose': { 'c1': { 'max_dim': 1200, 'orb_nfeatures': 4000, 'orb_scale_factor': 1.2, 'orb_nlevels': 8, 'match_ratio': 0.92, 'cluster_bin_size': 12.0, 'min_cluster_pairs': 5, 'min_keypoints': 10, 'min_area': 160, 'min_strength': 8, 'ocr_similarity': 0.4 }, 'c2': { 'canny_low': 50, 'canny_high': 150, 'edge_z': 1.4, 'stroke_z': 1.4, 'ocr_edge_density': 0.14, 'min_region_area': 60, 'min_line_components': 2, 'component_min_area': 60 }, 'c3': { 'component_min_area': 100, 'min_candidate_area': 150, 'stamp_circularity': 0.35, 'stamp_fill': 0.16, 'signature_aspect': 2.0, 'signature_fill_max': 0.42, 'fill_min': 0.25, 'ocr_min_area': 80, 'min_region_area': 80 }, 'c4': { 'gap_min_abs': 6.0, 'gap_median_mult': 1.0, 'gap_token_width_mult': 0.3, 'smooth_percentile': 60, 'smooth_min_area': 60, 'smooth_min_dim': 4, 'score_threshold': 0.9, 'min_region_area': 60, 'ring_var_ratio': 0.95, 'ring_grad_ratio': 0.92, 'ring_res_ratio': 0.95, 'ring_fg_delta': 0.01, 'erased_text_bonus': 0.85 }, 'c5': { 'row_density_thresh': 0.008, 'col_density_thresh': 0.004, 'gap_thresh_px': 7, 'gap_thresh_ratio': 0.035, 'band_min_height_px': 10, 'band_min_height_ratio': 0.035, 'band_min_width': 24, 'min_header_height_ratio': 0.05, 'cue_threshold': 0.15, 'cue_count_min': 2, 'score_threshold': 0.8, 'dist_balance_max': 0.32, 'dist_min': 0.35, 'min_region_area': 200 }, 'c6': { 'var_percentile': 50, 'grad_percentile': 50, 'res_percentile': 50, 'candidate_density_min': 0.0008, 'canny_low': 20, 'canny_high': 80, 'hough_threshold': 40, 'hough_min_len_ratio': 0.055, 'hough_min_len_px': 24, 'hough_max_gap': 24, 'diag_angle_min': 14.0, 'diag_angle_max': 76.0, 'diag_ratio_min': 0.18, 'diag_count_min': 3, 'periodicity_min': 1.5, 'contour_area_min': 70, 'aspect_min': 1.4, 'angle_delta_max': 50.0, 'box_area_min': 100, 'fallback_box_area_min': 320 }, 'c7': { 'min_tokens_page': 2, 'min_tokens_per_line': 2, 'min_gaps_per_line': 2, 'med_gap_min': 2.0, 'mad_gap_min': 0.6, 'gap_cv_max': 1.0, 'min_token_width': 3, 'large_gap_min_abs': 7.0, 'large_gap_median_mult': 1.3, 'large_gap_z': 1.8, 'large_gap_line_ratio_max': 0.4, 'tight_gap_median_mult': 0.65, 'tight_gap_median_min': 4.0, 'tight_gap_z': -1.4, 'single_gap_median_mult': 2.2, 'single_gap_min_abs': 9.0, 'single_gap_min_tokens': 3, 'pad_y_ratio': 0.25, 'pad_x_ratio': 0.1 }, 'c8': { 'text_density_min': 0.003, 'bg_var_ratio_max': 0.8, 'bg_res_ratio_max': 0.8, 'bg_grad_ratio_max': 0.9, 'stroke_cv_max': 0.36, 'height_cv_max': 0.27, 'spec_peak_min': 6.0, 'spec_highfreq_min': 1.4, 'spec_flatness_min': 0.4, 'bg_res_ratio_flatness_max': 0.9, 'coverage_min': 0.18, 'line_coverage_min': 0.3, 'coverage_high_min': 0.4, 'score_threshold': 1.7 }, 'c9': { 'canny_low': 40, 'canny_high': 120, 'min_token_area_line': 50, 'min_token_area_susp': 70, 'min_line_tokens': 2, 'z_edge': 1.6, 'z_grad': 1.6, 'z_res': 1.6, 'z_var': 1.6, 'z_stroke': 1.6, 'z_height': 1.8, 'ring_ratio_low': 0.9, 'ring_ratio_high': 1.2, 'fft_peak_min': 6.0, 'fft_ratio_min': 1.2, 'score_threshold': 1.6, 'min_region_area': 80, 'pad_x_ratio': 0.4, 'pad_y_ratio': 0.25, 'pad_x_min': 3, 'pad_y_min': 2 } } }

ACTIVE_TUNING = "super_loose"


def set_tuning_preset(name: str) -> None:
    global ACTIVE_TUNING
    if name not in DETECTOR_TUNING:
        raise KeyError(f"Unknown tuning preset: {name}")
    ACTIVE_TUNING = name


def _get_tuning() -> Dict[str, Any]:
    tuning = DETECTOR_TUNING.get(ACTIVE_TUNING)
    if tuning is None:
        tuning = DETECTOR_TUNING["current"]
    return tuning


# =========================
# DETECTORS
# =========================

def _cluster_copy_move_pairs(
    keypoints: List[Any],
    matches: List[Any],
    bin_size: float,
    min_pairs: int,
) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], int]]:
    if not matches:
        return []
    min_pairs = max(1, int(min_pairs))
    clusters: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for m in matches:
        src = keypoints[m.queryIdx].pt
        dst = keypoints[m.trainIdx].pt
        shift_x = dst[0] - src[0]
        shift_y = dst[1] - src[1]
        if abs(shift_x) < 3 and abs(shift_y) < 3:
            continue
        bin_key = (int(round(shift_x / bin_size)), int(round(shift_y / bin_size)))
        clusters.setdefault(bin_key, []).append((m.queryIdx, m.trainIdx))

    regions = []
    for _, pairs in clusters.items():
        if len(pairs) < min_pairs:
            continue
        src_pts = np.array([keypoints[i].pt for i, _ in pairs])
        dst_pts = np.array([keypoints[j].pt for _, j in pairs])
        src_box = (
            int(np.min(src_pts[:, 0])),
            int(np.min(src_pts[:, 1])),
            int(np.max(src_pts[:, 0])) + 1,
            int(np.max(src_pts[:, 1])) + 1,
        )
        dst_box = (
            int(np.min(dst_pts[:, 0])),
            int(np.min(dst_pts[:, 1])),
            int(np.max(dst_pts[:, 0])) + 1,
            int(np.max(dst_pts[:, 1])) + 1,
        )
        regions.append((src_box, dst_box, len(pairs)))
    return regions


def _detect_copy_move_candidates(
    gray: np.ndarray,
    mask: Optional[np.ndarray],
) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], int]]:
    if cv2 is None or gray is None:
        return []
    c1 = _get_tuning()["c1"]
    orb = cv2.ORB_create(
        nfeatures=int(c1["orb_nfeatures"]),
        scaleFactor=float(c1["orb_scale_factor"]),
        nlevels=int(c1["orb_nlevels"]),
    )
    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    if descriptors is None or len(keypoints) < int(c1["min_keypoints"]):
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(descriptors, descriptors, k=2)
    good = []
    ratio = float(c1["match_ratio"])
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.queryIdx == m.trainIdx:
            continue
        if m.distance < ratio * n.distance:
            good.append(m)
    return _cluster_copy_move_pairs(
        keypoints,
        good,
        bin_size=float(c1["cluster_bin_size"]),
        min_pairs=int(c1["min_cluster_pairs"]),
    )


def _c1_copy_move_regions(page: DocumentPage) -> List[DetectedRegion]:
    gray = _load_page_gray(page)
    if gray is None:
        return []
    c1 = _get_tuning()["c1"]
    gray_small, scale = _resize_max(gray, max_dim=int(c1["max_dim"]))
    mask = _text_mask(gray_small)
    candidates = _detect_copy_move_candidates(gray_small, mask)
    if not candidates:
        return []

    regions: List[DetectedRegion] = []
    height, width = gray.shape[:2]

    for src_box, dst_box, strength in candidates:
        src_box = _scale_box(src_box, scale)
        dst_box = _scale_box(dst_box, scale)
        src_box = _clip_box(src_box, width, height)
        dst_box = _clip_box(dst_box, width, height)
        if src_box is None or dst_box is None:
            continue
        src_area = (src_box[2] - src_box[0]) * (src_box[3] - src_box[1])
        dst_area = (dst_box[2] - dst_box[0]) * (dst_box[3] - dst_box[1])
        min_area = float(c1["min_area"])
        if src_area < min_area or dst_area < min_area or strength < int(c1["min_strength"]):
            continue
        ocr_sim = _ocr_similarity(gray, src_box, dst_box)
        if pytesseract is not None and ocr_sim < float(c1["ocr_similarity"]):
            continue
        chosen = dst_box
        regions.append(
            DetectedRegion(
                x=int(chosen[0]),
                y=int(chosen[1]),
                w=int(chosen[2] - chosen[0]),
                h=int(chosen[3] - chosen[1]),
                category_id="C1",
            )
        )

    return regions


def _c3_added_content_regions(page: DocumentPage) -> List[DetectedRegion]:
    if cv2 is None:
        return []
    gray = _load_page_gray(page)
    if gray is None:
        return []

    mask = _text_mask(gray)
    if mask is None:
        return []

    c3 = _get_tuning()["c3"]
    line_boxes = _extract_text_lines(mask)
    stats = _component_stats(mask, min_area=int(c3["component_min_area"]))
    ocr_boxes = _ocr_token_boxes(gray)

    candidates: List[Tuple[int, int, int, int]] = []

    for item in stats:
        box = item["box"]
        if _is_inside_any(box, line_boxes):
            continue
        area = item["area"]
        fill = item["fill"]
        circularity = item["circularity"]
        aspect = item["aspect"]
        if area < float(c3["min_candidate_area"]):
            continue

        is_stamp = circularity > float(c3["stamp_circularity"]) and fill > float(c3["stamp_fill"])
        is_signature = aspect > float(c3["signature_aspect"]) and fill < float(c3["signature_fill_max"])
        if is_stamp or is_signature or fill > float(c3["fill_min"]):
            candidates.append(box)

    for box in ocr_boxes:
        if _is_inside_any(box, line_boxes):
            continue
        x1, y1, x2, y2 = box
        if (x2 - x1) * (y2 - y1) >= float(c3["ocr_min_area"]):
            candidates.append(box)

    if not candidates:
        return []

    height, width = gray.shape[:2]
    regions: List[DetectedRegion] = []
    min_region_area = float(c3["min_region_area"])
    for box in candidates:
        clipped = _clip_box(box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        if (x2 - x1) * (y2 - y1) < min_region_area:
            continue
        regions.append(
            DetectedRegion(
                x=int(x1),
                y=int(y1),
                w=int(x2 - x1),
                h=int(y2 - y1),
                category_id="C3",
                type="text",
            )
        )

    return regions


def _band_boxes_from_mask(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    if mask is None:
        return []
    c5 = _get_tuning()["c5"]
    h, w = mask.shape[:2]
    row_density = (mask > 0).mean(axis=1)
    active = row_density > float(c5["row_density_thresh"])
    segments: List[Tuple[int, int]] = []
    start = None
    for idx, val in enumerate(active):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(active) - 1))
    if not segments:
        return []
    gap_thresh = max(
        int(c5["gap_thresh_px"]),
        int(float(c5["gap_thresh_ratio"]) * float(h)),
    )
    bands: List[Tuple[int, int]] = []
    cur_start, cur_end = segments[0]
    for seg_start, seg_end in segments[1:]:
        if seg_start - cur_end <= gap_thresh:
            cur_end = seg_end
        else:
            bands.append((cur_start, cur_end))
            cur_start, cur_end = seg_start, seg_end
    bands.append((cur_start, cur_end))

    boxes: List[Tuple[int, int, int, int]] = []
    min_height = max(
        int(c5["band_min_height_px"]),
        int(float(c5["band_min_height_ratio"]) * float(h)),
    )
    min_width = int(c5["band_min_width"])
    col_thresh = float(c5["col_density_thresh"])
    for y1, y2 in bands:
        cols = (mask[y1 : y2 + 1, :] > 0).mean(axis=0)
        xs = np.where(cols > col_thresh)[0]
        if xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        if (y2 - y1) < min_height or (x2 - x1) < min_width:
            continue
        boxes.append((x1, y1, x2 + 1, y2 + 1))
    boxes.sort(key=lambda item: item[1])
    return boxes


def _c5_merge_regions(page: DocumentPage) -> List[DetectedRegion]:
    if cv2 is None:
        return []
    gray = _load_page_gray(page)
    if gray is None:
        return []
    mask = _text_mask(gray)
    if mask is None or not np.any(mask > 0):
        return []
    c5 = _get_tuning()["c5"]
    bands = _band_boxes_from_mask(mask)
    if len(bands) < 2:
        return []
    top = bands[0]
    bottom = bands[-1]
    h, w = gray.shape[:2]
    min_header = int(float(c5["min_header_height_ratio"]) * float(h))
    if (top[3] - top[1]) < min_header or (bottom[3] - bottom[1]) < min_header:
        return []
    residual, variance, gradient = _local_texture_maps(gray)
    edges = cv2.Canny(gray, 50, 150)
    token_boxes = _ocr_token_boxes(gray)
    top_noise = _region_noise_profile(residual, variance, gradient, top)
    bot_noise = _region_noise_profile(residual, variance, gradient, bottom)
    top_text = _region_text_profile(mask, edges, token_boxes, top)
    bot_text = _region_text_profile(mask, edges, token_boxes, bottom)
    top_block = _block_energy(gray, top)
    bot_block = _block_energy(gray, bottom)
    noise_diff = _profile_diff(top_noise, bot_noise)
    text_diff = _profile_diff(top_text, bot_text)
    block_diff = 0.0
    if max(top_block, bot_block) > 1e-6:
        block_diff = abs(top_block - bot_block) / max(top_block, bot_block)
    cue_thresh = float(c5["cue_threshold"])
    cue_count = int(noise_diff > cue_thresh) + int(text_diff > cue_thresh) + int(block_diff > cue_thresh)
    score = (noise_diff * 1.4) + (text_diff * 1.0) + (block_diff * 0.8)
    if cue_count < int(c5["cue_count_min"]) or score < float(c5["score_threshold"]):
        return []
    full_box = (0, 0, w, h)
    full_noise = _region_noise_profile(residual, variance, gradient, full_box)
    full_text = _region_text_profile(mask, edges, token_boxes, full_box)
    full_block = _block_energy(gray, full_box)
    top_dist = _profile_distance(top_noise, full_noise) + _profile_distance(top_text, full_text)
    bot_dist = _profile_distance(bot_noise, full_noise) + _profile_distance(bot_text, full_text)
    if max(full_block, 1e-6) > 0:
        top_dist += abs(top_block - full_block) / max(full_block, 1e-6)
        bot_dist += abs(bot_block - full_block) / max(full_block, 1e-6)
    chosen: List[Tuple[Tuple[int, int, int, int], str]] = []
    if abs(top_dist - bot_dist) < float(c5["dist_balance_max"]) and max(top_dist, bot_dist) > float(c5["dist_min"]):
        chosen = [(top, "header"), (bottom, "body")]
    elif top_dist >= bot_dist:
        chosen = [(top, "header")]
    else:
        chosen = [(bottom, "body")]
    regions: List[DetectedRegion] = []
    min_region_area = float(c5["min_region_area"])
    for box, tag in chosen:
        clipped = _clip_box(box, w, h)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        if (x2 - x1) * (y2 - y1) < min_region_area:
            continue
        region = DetectedRegion(
            x=int(x1),
            y=int(y1),
            w=int(x2 - x1),
            h=int(y2 - y1),
            category_id="C5",
            type="merged_region",
        )
        if tag == "header":
            region.header_source = "inconsistent_header"
        else:
            region.body_source = "inconsistent_body"
        regions.append(region)
    return regions


def _c6_candidate_mask(
    residual: np.ndarray,
    variance: np.ndarray,
    gradient: np.ndarray,
) -> np.ndarray:
    c6 = _get_tuning()["c6"]
    var_thresh = float(np.percentile(variance, c6["var_percentile"]))
    grad_thresh = float(np.percentile(gradient, c6["grad_percentile"]))
    res_thresh = float(np.percentile(residual, c6["res_percentile"]))
    candidate = (
        (variance <= var_thresh)
        & (gradient <= grad_thresh)
        & (residual <= res_thresh)
    ).astype(np.uint8)
    candidate = candidate * 255
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    )
    return candidate


def _dominant_diagonal_angle(edges: np.ndarray) -> Tuple[Optional[float], float]:
    c6 = _get_tuning()["c6"]
    h, w = edges.shape[:2]
    min_len = max(
        int(c6["hough_min_len_px"]),
        int(float(c6["hough_min_len_ratio"]) * float(max(h, w))),
    )
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=int(c6["hough_threshold"]),
        minLineLength=min_len,
        maxLineGap=int(c6["hough_max_gap"]),
    )
    if lines is None:
        return None, 0.0
    diag_angles: List[float] = []
    total = 0
    ang_min = float(c6["diag_angle_min"])
    ang_max = float(c6["diag_angle_max"])
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if dx == 0.0 and dy == 0.0:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle > 90.0:
            angle = 180.0 - angle
        total += 1
        if ang_min <= angle <= ang_max:
            diag_angles.append(angle)
    if total == 0 or len(diag_angles) < int(c6["diag_count_min"]):
        return None, 0.0
    diag_ratio = float(len(diag_angles)) / float(total)
    if diag_ratio < float(c6["diag_ratio_min"]):
        return None, diag_ratio
    return float(np.median(diag_angles)), diag_ratio


def _c6_watermark_removal_regions(page: DocumentPage) -> List[DetectedRegion]:
    if cv2 is None:
        return []
    gray = _load_page_gray(page)
    if gray is None:
        return []
    c6 = _get_tuning()["c6"]
    residual, variance, gradient = _local_texture_maps(gray)
    candidate = _c6_candidate_mask(residual, variance, gradient)
    if float(np.mean(candidate > 0)) < float(c6["candidate_density_min"]):
        return []
    res_norm = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(res_norm, int(c6["canny_low"]), int(c6["canny_high"]))
    dominant_angle, diag_ratio = _dominant_diagonal_angle(edges)
    if dominant_angle is None:
        return []
    periodicity = _band_periodicity(candidate, dominant_angle)
    if periodicity < float(c6["periodicity_min"]):
        return []
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < float(c6["contour_area_min"]):
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw <= 0.0 or rh <= 0.0:
            continue
        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        if aspect < float(c6["aspect_min"]):
            continue
        angle = rect[2]
        if angle < -45.0:
            angle += 90.0
        angle = abs(angle)
        if abs(angle - dominant_angle) > float(c6["angle_delta_max"]):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < float(c6["box_area_min"]):
            continue
        boxes.append((x, y, x + w, y + h))
    height, width = gray.shape[:2]
    regions: List[DetectedRegion] = []
    if boxes:
        for box in boxes:
            clipped = _clip_box(box, width, height)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            regions.append(
                DetectedRegion(
                    x=int(x1),
                    y=int(y1),
                    w=int(x2 - x1),
                    h=int(y2 - y1),
                    category_id="C6",
                    type="watermark_removed",
                )
            )
    else:
        x, y, w, h = cv2.boundingRect(candidate)
        if w * h >= float(c6["fallback_box_area_min"]):
            clipped = _clip_box((x, y, x + w, y + h), width, height)
            if clipped is not None:
                x1, y1, x2, y2 = clipped
                regions.append(
                    DetectedRegion(
                        x=int(x1),
                        y=int(y1),
                        w=int(x2 - x1),
                        h=int(y2 - y1),
                        category_id="C6",
                        type="watermark_band",
                    )
                )
    return regions


def _c7_spacing_regions(page: DocumentPage) -> List[DetectedRegion]:
    if cv2 is None:
        return []
    gray = _load_page_gray(page)
    if gray is None:
        return []
    mask = _text_mask(gray)
    if mask is None:
        return []
    c7 = _get_tuning()["c7"]
    line_boxes = _extract_text_lines(mask)
    token_boxes = _ocr_token_boxes(gray)
    if len(token_boxes) < int(c7["min_tokens_page"]):
        return []
    height, width = gray.shape[:2]
    regions: List[DetectedRegion] = []
    seen: List[Tuple[int, int, int, int]] = []
    for line_box in line_boxes:
        lx1, ly1, lx2, ly2 = line_box
        line_tokens = []
        for bx in token_boxes:
            tx1, ty1, tx2, ty2 = bx
            cy = (ty1 + ty2) // 2
            if ly1 <= cy <= ly2:
                line_tokens.append(bx)
        if len(line_tokens) < int(c7["min_tokens_per_line"]):
            continue
        line_tokens.sort(key=lambda item: item[0])
        gaps: List[float] = []
        gap_info: List[Tuple[float, Tuple[int, int, int, int], Tuple[int, int, int, int]]] = []
        for idx in range(len(line_tokens) - 1):
            left = line_tokens[idx]
            right = line_tokens[idx + 1]
            gap = float(right[0] - left[2])
            if gap <= 0:
                continue
            gaps.append(gap)
            gap_info.append((gap, left, right))
        if len(gaps) < int(c7["min_gaps_per_line"]):
            continue
        gaps_np = np.array(gaps, dtype=np.float32)
        med_gap = float(np.median(gaps_np))
        if med_gap < float(c7["med_gap_min"]):
            continue
        mad_gap = float(np.median(np.abs(gaps_np - med_gap)))
        if mad_gap < float(c7["mad_gap_min"]):
            continue
        gap_cv = mad_gap / (med_gap + 1e-6)
        if gap_cv > float(c7["gap_cv_max"]):
            continue
        line_width = float(lx2 - lx1)
        if line_width <= 0:
            continue
        irregular: List[Tuple[float, Tuple[int, int, int, int], Tuple[int, int, int, int], bool]] = []
        min_tok_w = float(c7["min_token_width"])
        for gap, left, right in gap_info:
            left_w = left[2] - left[0]
            right_w = right[2] - right[0]
            if left_w < min_tok_w or right_w < min_tok_w:
                continue
            z = _robust_z(float(gap), gaps_np)
            is_large = (
                gap > max(float(c7["large_gap_min_abs"]), med_gap * float(c7["large_gap_median_mult"]))
                and z > float(c7["large_gap_z"])
                and gap < float(c7["large_gap_line_ratio_max"]) * line_width
            )
            is_tight = (
                gap < med_gap * float(c7["tight_gap_median_mult"])
                and med_gap > float(c7["tight_gap_median_min"])
                and z < float(c7["tight_gap_z"])
            )
            if not is_large and not is_tight:
                continue
            irregular.append((gap, left, right, is_large))
        if len(irregular) == 0:
            continue
        if len(irregular) < 2:
            gap, _, _, is_large = irregular[0]
            if not (
                is_large
                and gap > med_gap * float(c7["single_gap_median_mult"])
                and gap > float(c7["single_gap_min_abs"])
                and len(line_tokens) >= int(c7["single_gap_min_tokens"])
            ):
                continue
        pad_y = max(2, int(float(c7["pad_y_ratio"]) * max(1, ly2 - ly1)))
        for gap, left, right, _ in irregular:
            x1 = int(left[2])
            x2 = int(right[0])
            if x2 <= x1:
                continue
            pad_x = max(1, int(float(c7["pad_x_ratio"]) * max(1, x2 - x1)))
            box = (x1 - pad_x, ly1 - pad_y, x2 + pad_x, ly2 + pad_y)
            clipped = _clip_box(box, width, height)
            if clipped is None:
                continue
            if clipped in seen:
                continue
            seen.append(clipped)
            stretch = float(gap / (med_gap + 1e-6))
            regions.append(
                DetectedRegion(
                    x=int(clipped[0]),
                    y=int(clipped[1]),
                    w=int(clipped[2] - clipped[0]),
                    h=int(clipped[3] - clipped[1]),
                    category_id="C7",
                    type="irregular_spacing",
                    stretch_factor=stretch,
                )
            )
    return regions


def _c4_gap_candidates(
    gray: np.ndarray,
    mask: np.ndarray,
    line_boxes: List[Tuple[int, int, int, int]],
    token_boxes: List[Tuple[int, int, int, int]],
) -> List[Tuple[Tuple[int, int, int, int], str]]:
    candidates: List[Tuple[Tuple[int, int, int, int], str]] = []
    if not line_boxes or not token_boxes:
        return candidates

    c4 = _get_tuning()["c4"]

    for line_box in line_boxes:
        lx1, ly1, lx2, ly2 = line_box
        line_tokens = []
        for token_box in token_boxes:
            tx1, ty1, tx2, ty2 = token_box
            cy = (ty1 + ty2) // 2
            if ly1 <= cy <= ly2:
                line_tokens.append(token_box)

        if len(line_tokens) < 2:
            continue

        line_tokens.sort(key=lambda item: item[0])
        token_widths = [max(1, tx2 - tx1) for tx1, _, tx2, _ in line_tokens]
        median_token_width = float(np.median(token_widths)) if token_widths else 0.0
        gaps = []
        for idx in range(len(line_tokens) - 1):
            left = line_tokens[idx]
            right = line_tokens[idx + 1]
            gap_width = right[0] - left[2]
            if gap_width <= 0:
                continue
            gaps.append(gap_width)
        median_gap = float(np.median(gaps)) if gaps else 0.0

        for idx in range(len(line_tokens) - 1):
            left = line_tokens[idx]
            right = line_tokens[idx + 1]
            gap_width = right[0] - left[2]
            if gap_width <= 0:
                continue
            if gap_width < max(
                float(c4["gap_min_abs"]),
                median_gap * float(c4["gap_median_mult"]),
                median_token_width * float(c4["gap_token_width_mult"]),
            ):
                continue
            pad_y = max(2, (ly2 - ly1) // 10)
            candidate = (left[2], max(0, ly1 - pad_y), right[0], min(gray.shape[0], ly2 + pad_y))
            candidates.append((candidate, "erased_text"))

    return candidates


def _c4_smooth_candidates(gray: np.ndarray, mask: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], str]]:
    if cv2 is None:
        return []
    foreground = (mask > 0).astype(np.uint8)
    if not np.any(foreground):
        return []

    c4 = _get_tuning()["c4"]

    context_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
    content_context = cv2.dilate(foreground, context_kernel, iterations=1)
    interior = cv2.subtract(content_context, foreground)

    residual, variance, gradient = _local_texture_maps(gray)

    context_idx = content_context > 0
    if not np.any(context_idx):
        return []

    var_thresh = float(np.percentile(variance[context_idx], c4["smooth_percentile"]))
    grad_thresh = float(np.percentile(gradient[context_idx], c4["smooth_percentile"]))
    res_thresh = float(np.percentile(residual[context_idx], c4["smooth_percentile"]))

    candidate_mask = (
        (interior > 0)
        & (variance <= var_thresh)
        & (gradient <= grad_thresh)
        & (residual <= res_thresh)
    ).astype(np.uint8)

    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[Tuple[int, int, int, int], str]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < c4["smooth_min_area"]:
            continue
        if w < c4["smooth_min_dim"] or h < c4["smooth_min_dim"]:
            continue
        candidates.append(((x, y, x + w, y + h), "erased_content"))

    return candidates


def _score_c4_candidate(
    page: DocumentPage,
    gray: np.ndarray,
    residual: np.ndarray,
    variance: np.ndarray,
    gradient: np.ndarray,
    foreground: np.ndarray,
    box: Tuple[int, int, int, int],
    source: str,
) -> float:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return 0.0

    c4 = _get_tuning()["c4"]

    cand_var = _box_mean(variance, box)
    cand_grad = _box_mean(gradient, box)
    cand_res = _box_mean(residual, box)
    cand_fg = _box_mean(foreground, box)

    pad = max(12, min(28, max(x2 - x1, y2 - y1) // 2))
    ring_var = _box_mean_ring(variance, box, pad)
    ring_grad = _box_mean_ring(gradient, box, pad)
    ring_res = _box_mean_ring(residual, box, pad)
    ring_fg = _box_mean_ring(foreground, box, pad)

    score = 0.0
    if ring_var > 1e-6 and cand_var < ring_var * c4["ring_var_ratio"]:
        score += 1.0
    if ring_grad > 1e-6 and cand_grad < ring_grad * c4["ring_grad_ratio"]:
        score += 1.0
    if ring_res > 1e-6 and cand_res < ring_res * c4["ring_res_ratio"]:
        score += 0.75
    if ring_fg > cand_fg + c4["ring_fg_delta"]:
        score += 0.5
    if source == "erased_text":
        score += c4["erased_text_bonus"]
    score += _jpeg_block_bonus(page, gray, box)
    return score


def _c4_erased_regions(page: DocumentPage) -> List[DetectedRegion]:
    if cv2 is None:
        return []
    gray = _load_page_gray(page)
    if gray is None:
        return []

    mask = _text_mask(gray)
    if mask is None:
        return []

    c4 = _get_tuning()["c4"]

    foreground = (mask > 0).astype(np.uint8)
    residual, variance, gradient = _local_texture_maps(gray)
    line_boxes = _extract_text_lines(mask)
    token_boxes = _ocr_token_boxes(gray)

    candidate_specs: List[Tuple[Tuple[int, int, int, int], str]] = []
    candidate_specs.extend(_c4_gap_candidates(gray, mask, line_boxes, token_boxes))
    candidate_specs.extend(_c4_smooth_candidates(gray, mask))

    if not candidate_specs:
        return []

    height, width = gray.shape[:2]
    regions: List[DetectedRegion] = []
    seen_boxes: List[Tuple[int, int, int, int]] = []

    for box, source in candidate_specs:
        clipped = _clip_box(box, width, height)
        if clipped is None:
            continue
        if clipped in seen_boxes:
            continue
        seen_boxes.append(clipped)

        score = _score_c4_candidate(page, gray, residual, variance, gradient, foreground, clipped, source)
        if score < c4["score_threshold"]:
            continue

        x1, y1, x2, y2 = clipped
        if (x2 - x1) * (y2 - y1) < c4["min_region_area"]:
            continue
        regions.append(
            DetectedRegion(
                x=int(x1),
                y=int(y1),
                w=int(x2 - x1),
                h=int(y2 - y1),
                category_id="C4",
                type=source,
            )
        )

    return regions


def _c2_overwrite_regions(page: DocumentPage) -> List[DetectedRegion]:
    if cv2 is None:
        return []
    gray = _load_page_gray(page)
    if gray is None:
        return []

    mask = _text_mask(gray)
    if mask is None:
        return []

    c2 = _get_tuning()["c2"]

    edges = cv2.Canny(gray, c2["canny_low"], c2["canny_high"])
    text_pixels = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(text_pixels, cv2.DIST_L2, 3)

    line_boxes = _extract_text_lines(mask)
    comp_boxes = _component_boxes(mask, min_area=c2["component_min_area"])
    ocr_boxes = _ocr_token_boxes(gray)

    candidates: List[Tuple[int, int, int, int]] = []

    for line_box in line_boxes:
        lx1, ly1, lx2, ly2 = line_box
        line_comp = []
        for bx in comp_boxes:
            x1, y1, x2, y2 = bx
            cy = (y1 + y2) // 2
            if ly1 <= cy <= ly2:
                line_comp.append(bx)

        if len(line_comp) < c2["min_line_components"]:
            continue

        edge_vals = []
        stroke_vals = []
        for bx in line_comp:
            x1, y1, x2, y2 = bx
            if x2 <= x1 or y2 <= y1:
                continue
            box_edge = edges[y1:y2, x1:x2]
            box_text = text_pixels[y1:y2, x1:x2]
            box_dist = dist[y1:y2, x1:x2]
            edge_density = float(np.mean(box_edge > 0))
            stroke_var = float(np.var(box_dist[box_text > 0])) if np.any(box_text) else 0.0
            edge_vals.append(edge_density)
            stroke_vals.append(stroke_var)

        edge_vals = np.array(edge_vals, dtype=np.float32)
        stroke_vals = np.array(stroke_vals, dtype=np.float32)

        for idx, bx in enumerate(line_comp):
            edge_z = _robust_z(edge_vals[idx], edge_vals)
            stroke_z = _robust_z(stroke_vals[idx], stroke_vals)
            if edge_z > c2["edge_z"] or stroke_z > c2["stroke_z"]:
                candidates.append(bx)

    for bx in ocr_boxes:
        x1, y1, x2, y2 = bx
        height = y2 - y1
        width = x2 - x1
        if height <= 0 or width <= 0:
            continue
        local_edges = float(np.mean(edges[y1:y2, x1:x2] > 0))
        if local_edges > c2["ocr_edge_density"]:
            candidates.append(bx)

    if not candidates:
        return []

    height, width = gray.shape[:2]
    regions: List[DetectedRegion] = []
    for box in candidates:
        clipped = _clip_box(box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        if (x2 - x1) * (y2 - y1) < c2["min_region_area"]:
            continue
        regions.append(
            DetectedRegion(
                x=int(x1),
                y=int(y1),
                w=int(x2 - x1),
                h=int(y2 - y1),
                category_id="C2",
            )
        )

    return regions


def _c9_ai_edit_regions(page: DocumentPage) -> List[DetectedRegion]:
    if cv2 is None:
        return []
    gray = _load_page_gray(page)
    if gray is None:
        return []
    mask = _text_mask(gray)
    if mask is None:
        return []
    c9 = _get_tuning()["c9"]
    line_boxes = _extract_text_lines(mask)
    token_boxes = _ocr_token_boxes(gray)
    if not token_boxes:
        token_boxes = _component_boxes(mask, min_area=160)
    if not token_boxes or not line_boxes:
        return []
    edges = cv2.Canny(gray, int(c9["canny_low"]), int(c9["canny_high"]))
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    residual, variance, gradient = _local_texture_maps(gray)
    height, width = gray.shape[:2]
    regions: List[DetectedRegion] = []
    seen: List[Tuple[int, int, int, int]] = []
    for line_box in line_boxes:
        lx1, ly1, lx2, ly2 = line_box
        line_tokens = []
        for bx in token_boxes:
            tx1, ty1, tx2, ty2 = bx
            cy = (ty1 + ty2) // 2
            if ly1 <= cy <= ly2:
                area = (tx2 - tx1) * (ty2 - ty1)
                if area >= float(c9["min_token_area_line"]):
                    line_tokens.append(bx)
        if len(line_tokens) < int(c9["min_line_tokens"]):
            continue
        line_tokens.sort(key=lambda b: b[0])
        profiles: List[Dict[str, float]] = []
        for bx in line_tokens:
            profiles.append(_ai_box_profile(gray, mask, edges, residual, variance, gradient, dist, bx))
        edge_vals = np.array([p["edge"] for p in profiles], dtype=np.float32)
        grad_vals = np.array([p["grad"] for p in profiles], dtype=np.float32)
        res_vals = np.array([p["res"] for p in profiles], dtype=np.float32)
        var_vals = np.array([p["var"] for p in profiles], dtype=np.float32)
        stroke_vals = np.array([p["stroke"] for p in profiles], dtype=np.float32)
        h_vals = np.array([p["h"] for p in profiles], dtype=np.float32)
        w_vals = np.array([p["w"] for p in profiles], dtype=np.float32)
        fft_ratio_vals = np.array([p["fft_ratio"] for p in profiles], dtype=np.float32)
        fft_peak_vals = np.array([p["fft_peak"] for p in profiles], dtype=np.float32)
        gaps = []
        for idx in range(len(line_tokens) - 1):
            gap = float(line_tokens[idx + 1][0] - line_tokens[idx][2])
            if gap > 0:
                gaps.append(gap)
        med_gap = float(np.median(gaps)) if gaps else 6.0
        pad_x = max(int(c9["pad_x_min"]), int(float(c9["pad_x_ratio"]) * med_gap))
        pad_y = max(int(c9["pad_y_min"]), int(float(c9["pad_y_ratio"]) * max(1, ly2 - ly1)))
        suspicious: List[Tuple[int, int, int, int]] = []
        for idx, bx in enumerate(line_tokens):
            tx1, ty1, tx2, ty2 = bx
            if (tx2 - tx1) * (ty2 - ty1) < float(c9["min_token_area_susp"]):
                continue
            score = 0.0
            z_edge = abs(_robust_z(edge_vals[idx], edge_vals))
            z_grad = abs(_robust_z(grad_vals[idx], grad_vals))
            z_res = abs(_robust_z(res_vals[idx], res_vals))
            z_var = abs(_robust_z(var_vals[idx], var_vals))
            z_stroke = abs(_robust_z(stroke_vals[idx], stroke_vals))
            z_h = abs(_robust_z(h_vals[idx], h_vals))
            if z_edge > float(c9["z_edge"]):
                score += 1.0
            if z_grad > float(c9["z_grad"]):
                score += 1.0
            if z_res > float(c9["z_res"]):
                score += 1.0
            if z_var > float(c9["z_var"]):
                score += 1.0
            if z_stroke > float(c9["z_stroke"]):
                score += 0.7
            if z_h > float(c9["z_height"]):
                score += 0.6
            pad = max(6, int(0.35 * min(tx2 - tx1, ty2 - ty1)))
            res_ratio = _box_ring_ratio(residual, bx, pad)
            var_ratio = _box_ring_ratio(variance, bx, pad)
            grad_ratio = _box_ring_ratio(gradient, bx, pad)
            ring_low = float(c9["ring_ratio_low"])
            ring_high = float(c9["ring_ratio_high"])
            ring_hits = int(res_ratio < ring_low or res_ratio > ring_high)
            ring_hits += int(var_ratio < ring_low or var_ratio > ring_high)
            ring_hits += int(grad_ratio < ring_low or grad_ratio > ring_high)
            if ring_hits >= 2:
                score += 1.0
            if fft_peak_vals[idx] > float(c9["fft_peak_min"]):
                score += 0.8
            if fft_ratio_vals[idx] > float(c9["fft_ratio_min"]):
                score += 0.6
            if score >= float(c9["score_threshold"]):
                suspicious.append(bx)
        if not suspicious:
            continue
        merged = _merge_line_boxes(suspicious, pad_x, pad_y)
        for box in merged:
            clipped = _clip_box(box, width, height)
            if clipped is None:
                continue
            if clipped in seen:
                continue
            seen.append(clipped)
            x1, y1, x2, y2 = clipped
            if (x2 - x1) * (y2 - y1) < float(c9["min_region_area"]):
                continue
            regions.append(
                DetectedRegion(
                    x=int(x1),
                    y=int(y1),
                    w=int(x2 - x1),
                    h=int(y2 - y1),
                    category_id="C9",
                    type="ai_edit",
                )
            )
    return regions


def _c8_ai_document(
    page: DocumentPage,
    c9_regions: Optional[List[DetectedRegion]] = None,
) -> Tuple[bool, float]:
    if cv2 is None:
        return False, 0.0
    gray = _load_page_gray(page)
    if gray is None:
        return False, 0.0
    mask = _text_mask(gray)
    if mask is None:
        return False, 0.0
    c8 = _get_tuning()["c8"]
    text_density = float(np.mean(mask > 0))
    if text_density < float(c8["text_density_min"]):
        return False, 0.0
    residual, variance, gradient = _local_texture_maps(gray)
    bg_idx = mask == 0
    eps = 1e-6
    if np.any(bg_idx):
        bg_var = float(np.mean(variance[bg_idx]))
        bg_res = float(np.mean(residual[bg_idx]))
        bg_grad = float(np.mean(gradient[bg_idx]))
    else:
        bg_var = float(np.mean(variance))
        bg_res = float(np.mean(residual))
        bg_grad = float(np.mean(gradient))
    global_var = float(np.mean(variance)) + eps
    global_res = float(np.mean(residual)) + eps
    global_grad = float(np.mean(gradient)) + eps
    bg_var_ratio = bg_var / global_var
    bg_res_ratio = bg_res / global_res
    bg_grad_ratio = bg_grad / global_grad
    token_boxes = _ocr_token_boxes(gray)
    if not token_boxes:
        token_boxes = _component_boxes(mask, min_area=160)
    stroke_cv = None
    height_cv = None
    if token_boxes:
        dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        strokes = []
        heights = []
        for bx in token_boxes:
            x1, y1, x2, y2 = bx
            area = (x2 - x1) * (y2 - y1)
            if area < 120:
                continue
            heights.append(max(1.0, float(y2 - y1)))
            strokes.append(max(1e-6, _box_stroke_mean(mask, dist, bx)))
        if len(heights) >= 6 and len(strokes) >= 6:
            height_cv = float(np.std(heights) / (np.mean(heights) + eps))
            stroke_cv = float(np.std(strokes) / (np.mean(strokes) + eps))
    spec = _fft_spectrum_features(gray)
    line_boxes = _extract_text_lines(mask)
    if c9_regions is None:
        c9_regions = []
    coverage, line_coverage = _ai_region_coverage(c9_regions, mask, line_boxes)
    score = 0.0
    if bg_var_ratio < float(c8["bg_var_ratio_max"]) and bg_res_ratio < float(c8["bg_res_ratio_max"]):
        score += 1.0
    if bg_grad_ratio < float(c8["bg_grad_ratio_max"]):
        score += 0.6
    if stroke_cv is not None and height_cv is not None:
        if stroke_cv < float(c8["stroke_cv_max"]) and height_cv < float(c8["height_cv_max"]):
            score += 0.9
    if spec["peak_ratio"] > float(c8["spec_peak_min"]):
        score += 0.9
    if spec["highfreq_ratio"] > float(c8["spec_highfreq_min"]):
        score += 0.7
    if spec["flatness"] > float(c8["spec_flatness_min"]) and bg_res_ratio < float(c8["bg_res_ratio_flatness_max"]):
        score += 0.6
    if coverage > float(c8["coverage_min"]) and line_coverage > float(c8["line_coverage_min"]):
        score += 1.2
    if coverage > float(c8["coverage_high_min"]):
        score += 0.8
    return score >= float(c8["score_threshold"]), score


# =========================
# PIPELINE FALLBACKS + QUALITY (ADD-ON)
# =========================

def detect_tampering_categories_sync(page: DocumentPage) -> List[str]:
    predicted: List[str] = []
    c1_regions = _c1_copy_move_regions(page)
    if c1_regions:
        predicted.append("C1")
    c2_regions = _c2_overwrite_regions(page)
    if c2_regions:
        predicted.append("C2")
    c3_regions = _c3_added_content_regions(page)
    if c3_regions:
        predicted.append("C3")
    c4_regions = _c4_erased_regions(page)
    if c4_regions:
        predicted.append("C4")
    c5_regions = _c5_merge_regions(page)
    if c5_regions:
        predicted.append("C5")
    c6_regions = _c6_watermark_removal_regions(page)
    if c6_regions:
        predicted.append("C6")
    c7_regions = _c7_spacing_regions(page)
    if c7_regions:
        predicted.append("C7")
    c9_regions = _c9_ai_edit_regions(page)
    if c9_regions:
        predicted.append("C9")
    c8_flag, _ = _c8_ai_document(page, c9_regions=c9_regions)
    if c8_flag and (not predicted or predicted == ["C9"]):
        return ["C8"]
    return predicted or ["C10"]


def localize_tampered_regions_sync(
    page: DocumentPage,
    predicted_categories: List[str],
) -> List[DetectedRegion]:
    regions: List[DetectedRegion] = []
    if "C1" in predicted_categories:
        regions.extend(_c1_copy_move_regions(page))
    if "C2" in predicted_categories:
        regions.extend(_c2_overwrite_regions(page))
    if "C3" in predicted_categories:
        regions.extend(_c3_added_content_regions(page))
    if "C4" in predicted_categories:
        regions.extend(_c4_erased_regions(page))
    if "C5" in predicted_categories:
        regions.extend(_c5_merge_regions(page))
    if "C6" in predicted_categories:
        regions.extend(_c6_watermark_removal_regions(page))
    if "C7" in predicted_categories:
        regions.extend(_c7_spacing_regions(page))
    if "C9" in predicted_categories:
        regions.extend(_c9_ai_edit_regions(page))
    return regions


def assess_page_quality(page: DocumentPage) -> Dict[str, Any]:
    gray = _load_page_gray(page)
    if gray is None:
        return fallback_quality_summary(page)
    h, w = gray.shape[:2]
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    lap_var = 0.0
    edge_density = 0.0
    if cv2 is not None:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(lap.var())
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges > 0))
        blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
        residual = gray.astype(np.float32) - blur.astype(np.float32)
    else:
        residual = gray.astype(np.float32) - mean_val
    noise_mad = float(np.median(np.abs(residual)))
    mask = _text_mask(gray)
    text_density = float(np.mean(mask > 0)) if mask is not None else 0.0
    warnings: List[str] = []
    if lap_var > 0.0 and lap_var < 35.0:
        warnings.append("blurry")
    if std_val < 20.0:
        warnings.append("low_contrast")
    if mean_val < 50.0:
        warnings.append("too_dark")
    if mean_val > 210.0:
        warnings.append("too_bright")
    if text_density < 0.005:
        warnings.append("low_text")
    return {
        "ok": len(warnings) == 0,
        "mean": mean_val,
        "std": std_val,
        "lap_var": lap_var,
        "edge_density": edge_density,
        "noise_mad": noise_mad,
        "text_density": text_density,
        "warnings": warnings,
        "image_size": [int(w), int(h)],
    }


def fallback_quality_summary(page: DocumentPage) -> Dict[str, Any]:
    width = int(page.image_width) if page.image_width else None
    height = int(page.image_height) if page.image_height else None
    return {
        "ok": False,
        "reason": "quality_check_failed",
        "image_path": page.image_path or page.original_path,
        "image_size": [width, height],
    }


def fallback_predicted_categories(page: DocumentPage) -> List[str]:
    return ["C10"]


def fallback_regions(page: DocumentPage, predicted_categories: List[str]) -> List[DetectedRegion]:
    return []


def fallback_validation(
    page: DocumentPage,
    predicted_categories: List[str],
    regions: List[DetectedRegion],
) -> Dict[str, Any]:
    try:
        return validate_prediction(page, predicted_categories, regions)
    except Exception:
        return {"ok": False, "categories": predicted_categories or ["C10"], "requires_yaml": False}


def fallback_build_result(
    page: DocumentPage,
    predicted_categories: List[str],
    regions: List[DetectedRegion],
    quality_summary: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
) -> PageAnalysisResult:
    return PageAnalysisResult(
        source_link=page.source_link,
        file_name=page.page_file_name,
        original_file_name=page.original_file_name,
        page_number=page.page_number,
        predicted_categories=predicted_categories or ["C10"],
        detected_regions=regions or [],
        notes={"quality": quality_summary or {}, "validation": validation or {}},
    )


# =========================
# OUTPUT / EXPORT HELPERS
# =========================

def normalize_category_list(predicted_categories: List[str]) -> List[str]:
    valid = [c for c in predicted_categories if c in CATEGORY_IDS]
    seen = set()
    ordered = []
    for c in valid:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def build_json_row(result: PageAnalysisResult) -> Dict[str, Any]:
    categories = normalize_category_list(result.predicted_categories)
    category_string = "||".join(categories) if categories else "C10"

    return {
        "link": result.source_link,
        "file_name": result.file_name,
        "Category_ID": category_string,
    }


def detected_region_to_yaml_item(region: DetectedRegion) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "h": int(region.h),
        "w": int(region.w),
        "x": int(region.x),
        "y": int(region.y),
        "category_id": region.category_id,
    }

    if region.type is not None:
        item["type"] = region.type
    if region.stretch_factor is not None:
        item["stretch_factor"] = region.stretch_factor
    if region.header_source is not None:
        item["header_source"] = region.header_source
    if region.body_source is not None:
        item["body_source"] = region.body_source

    return item


def yaml_required_for_result(result: PageAnalysisResult) -> bool:
    categories = set(normalize_category_list(result.predicted_categories))
    if not categories:
        return False
    return not categories.issubset(CATEGORY_ONLY_CLASSES)


def yaml_name_for_page_file(page_file_name: str) -> str:
    return f"{Path(page_file_name).stem}.yaml"


def write_yaml_for_result(result: PageAnalysisResult, annotations_dir: Path) -> Optional[Path]:
    if not yaml_required_for_result(result):
        return None

    if yaml is None:
        raise ImportError("PyYAML is required to write annotation YAML files.")

    yaml_path = annotations_dir / yaml_name_for_page_file(result.file_name)
    yaml_items = [detected_region_to_yaml_item(region) for region in result.detected_regions]

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_items, f, sort_keys=False, allow_unicode=True)

    return yaml_path


def write_submission_json(results: List[PageAnalysisResult], output_dir: Path) -> Path:
    json_rows = [build_json_row(r) for r in results]
    json_path = output_dir / "submission.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2, ensure_ascii=False)

    return json_path


def write_optional_excel_summary(results: List[PageAnalysisResult], output_dir: Path) -> Optional[Path]:
    if pd is None:
        return None

    rows = []
    for r in results:
        rows.append({
            "link": r.source_link,
            "file_name": r.file_name,
            "Category_ID": "||".join(normalize_category_list(r.predicted_categories)) if r.predicted_categories else "C10",
            "region_count": len(r.detected_regions),
            "page_number": r.page_number,
            "original_file_name": r.original_file_name,
        })

    df = pd.DataFrame(rows)
    excel_path = output_dir / "submission_preview.xlsx"
    df.to_excel(excel_path, index=False)
    return excel_path


def export_all_outputs(results: List[PageAnalysisResult], output_dir: Path, annotations_dir: Path) -> Dict[str, Any]:
    annotation_paths = []
    for result in results:
        yaml_path = write_yaml_for_result(result, annotations_dir)
        if yaml_path is not None:
            annotation_paths.append(str(yaml_path))

    json_path = write_submission_json(results, output_dir)
    excel_path = write_optional_excel_summary(results, output_dir)

    return {
        "json_path": str(json_path),
        "yaml_paths": annotation_paths,
        "excel_preview_path": str(excel_path) if excel_path else None,
        "total_results": len(results),
    }


# =========================
# PIPELINE RUNNER
# =========================

def run_pipeline(
    input_path: Path,
    work_dir: Path,
    preset: Optional[str] = None,
    enable_ocr: Optional[bool] = None,
) -> Dict[str, Any]:
    if preset:
        set_tuning_preset(preset)
    if enable_ocr is not None:
        set_ocr_enabled(enable_ocr)

    input_dir = work_dir / "input"
    output_dir = work_dir / "output"
    annotations_dir = output_dir / "annotations"
    debug_dir = output_dir / "debug"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        dest = input_dir / input_path.name
        if dest != input_path:
            shutil.copy2(input_path, dest)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    render_dir = debug_dir / "rendered_pages"
    pages = build_document_pages(input_dir, render_dir)

    results: List[PageAnalysisResult] = []
    for page in pages:
        try:
            quality_summary = assess_page_quality(page)
            if quality_summary is None:
                quality_summary = fallback_quality_summary(page)
        except Exception:
            quality_summary = fallback_quality_summary(page)

        try:
            predicted_categories = detect_tampering_categories_sync(page)
            if predicted_categories is None:
                predicted_categories = fallback_predicted_categories(page)
        except Exception:
            predicted_categories = fallback_predicted_categories(page)

        predicted_categories = normalize_category_list(predicted_categories) or ["C10"]

        try:
            regions = localize_tampered_regions_sync(page, predicted_categories)
            if regions is None:
                regions = fallback_regions(page, predicted_categories)
        except Exception:
            regions = fallback_regions(page, predicted_categories)

        try:
            regions = postprocess_regions(regions, predicted_categories, page)
            if regions is None:
                regions = fallback_regions(page, predicted_categories)
        except Exception:
            regions = fallback_regions(page, predicted_categories)

        try:
            regions = enrich_region_metadata(regions, predicted_categories, page)
            if regions is None:
                regions = fallback_regions(page, predicted_categories)
        except Exception:
            regions = fallback_regions(page, predicted_categories)

        try:
            validation = validate_prediction(page, predicted_categories, regions)
            if validation is None:
                validation = fallback_validation(page, predicted_categories, regions)
        except Exception:
            validation = fallback_validation(page, predicted_categories, regions)

        try:
            result = build_page_analysis_result(
                page=page,
                predicted_categories=predicted_categories,
                regions=regions,
                quality_summary=quality_summary,
                validation=validation,
            )
            if result is None:
                result = fallback_build_result(page, predicted_categories, regions, quality_summary, validation)
        except Exception:
            result = fallback_build_result(page, predicted_categories, regions, quality_summary, validation)

        results.append(result)

    export_info = export_all_outputs(results, output_dir, annotations_dir)

    return {
        "pages": pages,
        "results": results,
        "export_info": export_info,
        "output_dir": output_dir,
        "annotations_dir": annotations_dir,
    }
