# 🔍 NHA PS3 - Document Forgery & Deepfake Detection
**Nikhileswara Rao Sulake<sup>1</sup>, Sai Manikanta Eswar Machara<sup>1</sup>, Sivalal Kethavath<sup>1</sup>**

<sup>1</sup> Rajiv Gandhi University of Knowledge Technologies, Nuzvid, Andhra Pradesh

---


### Internal Validation Score: **0.5315** &nbsp;·&nbsp; Leaderboard Position: **Top 3**

---

This pipeline is a **completely classical computer vision–based solution**, where no models or other heavy components are required. Just using traditional, pre-defined libraries, our solution is capable enough to detect document forgery across all 9 tampering categories.

**Why did we choose this classical CV approach over the big guns, CLIP, ViT, ManTra-Net, MVSS-Net, CAT-Net, or DTD?**

Because our main motive is making the solution **highly scalable, computationally lightweight, and 100% explainable**, aligning perfectly with the core demands of the AB-PMJAY ecosystem. These are government health insurance claim documents. They need to be processed at scale, across thousands of hospitals, on infrastructure that may not have GPUs. And when you flag a document as forged, you need to *explain why*, not just point to a confidence score from a black-box model.

| | Deep Learning Approaches | **Our Classical CV Pipeline** |
|---|---|---|
| **Scalability** | GPU-bound, memory-heavy | ✅ Runs on any CPU, anywhere |
| **Explainability** | Black-box, "the model says so" | ✅ Every flag traces back to a math equation |
| **Deployment** | Needs ONNX/TensorRT, model hosting | ✅ Single notebook, zero model files |
| **Latency** | Seconds per image even on GPU | ✅ ~1–3s per page on plain CPU |
| **AB-PMJAY Fit** | Over-engineered for document scans | ✅ Purpose-built for Indian medical docs |

> If you can explain *why* a region is tampered using math, you don't need a neural network to do it for you.

---
## Instructions to try our solution


### 📁 Project Structure

```
NHA PS3 Final Updated/
├── nha_ps3_skeletal_notebook_main.ipynb            # ← Main solution notebook (run this)
├── README.md                          # This file
├── output/
│   ├── output.json                    # Final JSON submission
│   └── ***.yaml                       # Per-page YAML bounding boxes
└── 1ae9a4db-.../Claim_Documents/      # Input dataset
```

---

### 🚀 How to Run the Code

Change the input directory path to pointing towards the folder containing all the health reports in PDFs, JPEGs, JPGs formats.
You can change the path using this variable in the second code cell: `INPUT_DIR`

And just click run all button in the notebook, that's it. The code will take care of it, we also show the results processing using a TQDM bar for better visualization of when will the process will be completed.


### 🌐 How to Run the Website Locally

We will set the local API URL to 127.0.0.1 on 8000 port, we can do any toggle in the configs in the respective config files in frontend and backend. Now, open the powershell and to first start the backend from the repo root, run the below code:

```
cd .\backend\
$env:DATA_DIR="$PWD\data"
$env:CORS_ORIGINS="http://127.0.0.1:5500,http://localhost:5500"
$env:AUTH_REQUIRED="false"
$env:CLOUDINARY_ENABLED="false"
uvicorn app.main:app --reload
```

To start the frontend from the repo root, run the below code in a new powershell:

```
cd .\frontend\
python -m http.server 5500
```

Then you can open the http://127.0.0.1:5500/app.html to see the website, and run all the things properly. Mentioning that the code will run purely on your CPU.


---

## 🔬 Per-Class Detection Methodology

### C1, Copy-Paste Detection

**Signal:** Duplicate regions within the same document displaced spatially.

```
Grayscale Image
  → ORB Keypoints (4000 features)
  → BFMatcher Self-Matching (k=2)
  → Lowe's Ratio Test (m.dist < 0.75·n.dist)
  → Spatial Shift Filter (‖shift‖ > min_px)
  → Shift-Vector Clustering (bin_size = 12px)
  → Cluster ≥ N pairs → Bounding Boxes
```

**Core Math, Shift-Vector Clustering:**

$$\vec{s}_i = \vec{p}_{\text{dst}} - \vec{p}_{\text{src}}, \quad \text{bin}(i) = \left(\left\lfloor \frac{s_x}{b} \right\rceil,\; \left\lfloor \frac{s_y}{b} \right\rceil\right)$$

Clusters with $|\text{bin}| \geq N_{\min}$ indicate copy-paste with consistent translation.

---

### C2, Overwrite Detection

**Signal:** Text components within a line that show anomalous edge density, stroke width, or ink darkness relative to their neighbours.

```
Per-Line Text Components
  ├── Edge Density (Canny)
  ├── Stroke Width (Distance Transform)
  └── Ink Darkness (mean intensity)
        ↓
  Robust Z-Score per feature
        ↓
  Z > threshold on ≥2 signals? → Yes → Flag as Overwritten
```

**Core Math, MAD-Based Robust Z-Score:**

$$Z_{\text{MAD}} = \frac{0.6745 \cdot (x_i - \tilde{x})}{\text{MAD}}, \quad \text{MAD} = \text{median}(|x_j - \tilde{x}|)$$

A component is flagged only when **≥ 2 signals** exceed their Z-thresholds simultaneously, with a hard per-page cap to prevent mass false positives.

---

### C3, Added Content Detection

**Signal:** Stamps (red/blue), signatures, and pasted elements that sit outside normal text lines.

```
Engine 1 (Color):                    Engine 2 (Shape):
  Color Image (BGR)                    Grayscale
  → HSV Conversion                     → Component Stats
  → Red Mask (H∈[0,10]∪[160,180])      → Circularity > τ ? → Stamp
  → Blue Mask (H∈[100,130])            → Aspect > τ ?      → Signature
  → Morphological Clean-up
  → Contour Extraction
  → Stamp / Seal Bounding Boxes
```

**Two-Engine Approach:**
1. **HSV Color Segmentation**, isolates red/blue ink artifacts common in Indian medical documents
2. **Shape Geometry**, circularity for round stamps, high aspect ratio for signatures

---

### C4, Erasure Detection

**Signal:** Gaps within text lines that are unnaturally smooth compared to surrounding content.

```
Text Lines
  → Detect Gaps (token spacing)
  → Gap > median × τ AND Gap > min_abs?
      ↓ Yes
  → Measure Gap Noise & Gradient
  → Compare vs Context Ring
  → noise_gap/noise_ctx < 0.3 AND grad_gap/grad_ctx < 0.4?
      ↓ Yes
  → Flag as Erasure
```

**Core Principle:**

$$\text{Score} = \mathbb{1}\!\left[\frac{\bar{N}_{\text{gap}}}{\bar{N}_{\text{ctx}}} < \tau_n\right] \cdot 2 + \mathbb{1}\!\left[\frac{\bar{G}_{\text{gap}}}{\bar{G}_{\text{ctx}}} < \tau_g\right] \cdot 2 + \mathbb{1}\!\left[\sigma^2_{\text{gap}} < 100\right]$$

Digitally erased regions lack the natural noise floor present in scanned paper, this ratio reliably separates real whitespace from artificial erasure.

---

### C5, Document Merge Detection

**Signal:** Header and body originate from different physical documents with distinct noise fingerprints.

```
Page Image
  → Segment into Horizontal Bands
  → Per-Band Noise Fingerprint
  → Compare Adjacent Band Profiles
  → Profile Distance > threshold? → Yes → Flag Merge Boundary
```

**Noise Fingerprint Vector:**

$$\vec{f}_b = \left[\;\bar{N}_b,\; \sigma_{N_b},\; \bar{G}_b,\; \bar{\mu}_b,\; \sigma_b\;\right]$$

$$d(b_i, b_j) = \frac{1}{K} \sum_{k=1}^{K} \frac{|f_{b_i}^{(k)} - f_{b_j}^{(k)}|}{|f_{b_i}^{(k)}| + |f_{b_j}^{(k)}| + \epsilon}$$

When $d(\text{top}, \text{bottom}) > \tau$, the page is flagged as a merge of two different source documents.

---

### C6, Watermark Removal Detection

**Signal:** Removed watermarks leave frequency-domain ghosts and unnaturally smooth background.

```
Grayscale Image
  ├── FFT Radial Power Spectrum → Autocorrelation Peak Count
  ├── CLAHE Enhancement         → Ghost Region Extraction
  └── Background Variance Map   → Low-Variance Anomaly Ratio
        ↓
  Combined Score ≥ τ? → Yes → Flag Watermark Removal
```

**Three complementary signals scored additively:**
| Signal | What it catches | Points |
|---|---|---|
| FFT Periodicity Peaks | Residual repeating patterns from watermark grid | 0.8–1.5 |
| CLAHE Ghost Detection | Faint remnants invisible to naked eye | 1.0 |
| Background Smoothness | Over-smoothed regions where watermark was painted over | 1.0 |

---

### C7, Irregular Spacing Detection

**Signal:** Statistically anomalous inter-word or inter-line gaps within OCR-extracted text.

```
OCR Token Boxes
  → Group by Text Line
  → Compute Inter-Word Gaps
  → Global Median & MAD
  → Per-Gap Z-Score
  → Z > τ_large OR Z < τ_tight? → Yes → Flag Irregular Spacing
```

### **Anomaly Criteria (three checks per gap)**

| Check | Condition | Catches |
|---|---|---|
| Large Gap | $Z > \tau_{large}$ AND $g > \tau_{abs}$ AND $g > \tau_{med} \cdot \tilde{g}_{line}$ | Inserted whitespace |
| Tight Gap | $Z < \tau_{tight}$ AND $\tilde{g}_{line} > 3$ | Compressed text |
| Extreme Gap | $g > \tau_{singleabs}$ AND $g > \tau_{singlemed} \cdot \tilde{g}_{line}$ | Single huge anomaly |


---

### C8, Fully AI-Generated Document

**Signal:** Multi-signal scoring across spectral, noise, texture, and typographic domains.

```
┌─ Spectral Analysis ──────────────────────────┐
│  FFT Radial Spectrum → Spectral Flatness     │
│                      → HF Energy Ratio       │
│                      → GAN Peak Count        │
├─ Noise Floor ────────────────────────────────┤
│  Background Noise Residual → Variance < τ ?  │
│                            → Kurtosis Check  │
├─ Typography ─────────────────────────────────┤
│  OCR Token Heights → Height CV < τ ?         │
│                    → Stroke Width CV < τ ?   │
├─ Texture ────────────────────────────────────┤
│  LBP Histogram → Entropy < τ ?               │
└──────────────────────────────────────────────┘
        ↓ All signals summed
  Total Score ≥ τ? → Yes → Category: C8
```

> **Category-only**, no bounding boxes required. Returns `C8` if total evidence score crosses threshold.

---

### C9, Partial AI Edits

**Signal:** Individual text components with anomalous noise, edge, gradient, or ELA profiles relative to their line context.

```
Per-Line Components
  ├── Noise Residual per Component
  ├── Edge Density per Component
  ├── Gradient Energy per Component
  └── ELA Level per Component
        ↓
  MAD Z-Score vs Line Peers
        ↓
  Score ≥ τ on ≥2 signals? → Yes → Flag AI-Edited Field
```

**ELA (Error Level Analysis):**

$$\text{ELA}(x,y) = |I(x,y) - I_{\text{recomp}}(x,y)|, \quad I_{\text{recomp}} = \text{JPEG}(I, Q{=}90)$$

Regions edited after initial compression show different ELA levels from surrounding authentic content.

**Safety Rail:** If $>\,20\%$ of all components on a page are flagged, the entire page result is suppressed (likely systematic noise, not targeted AI edits).

---

## ⚙️ Tuning System

All thresholds are **decoupled from detection logic** via a preset system. No code changes needed, just switch presets:

```python
set_preset("normal")       # Precision-first, tight thresholds
set_preset("loose")        # Balanced
set_preset("very_loose")   # Recall-first
set_preset("ultra_loose")  # Aggressive detection
set_preset("super_loose")  # Maximum recall, lower precision
```

Each preset controls **all 9 classes independently**, Z-score thresholds, minimum areas, per-page caps, and scoring weights are all parameterized per class.

---

## 🛡️ Robustness Design

| Feature | Implementation |
|---|---|
| **Crash-proof** | Every detector wrapped in `try/except` with `fallback_` functions |
| **Per-page caps** | Prevents mass false positives (max N detections per page) |
| **IoU deduplication** | Removes overlapping boxes within same category (IoU > 0.7) |
| **Quality gating** | Skips unreadable/blurry pages via Laplacian variance check |
| **Graceful degradation** | Missing Tesseract → OCR-dependent classes silently skip |

---

<p align="center">
  <b>Built for NHA Hackathon PS-03 &nbsp;·&nbsp; Zero Models &nbsp;·&nbsp; Pure Math &nbsp;·&nbsp; Full Explainability</b>
</p>
