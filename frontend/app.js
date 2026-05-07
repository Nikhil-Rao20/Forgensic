import {
  API_BASE_URL,
  FIREBASE_CONFIG,
  APP_OPTIONS,
  APP_BRAND,
  ADMIN_CREDENTIALS,
  CATEGORY_COLORS
} from "./config.js";
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.4/firebase-app.js";
import {
  getAuth,
  onAuthStateChanged,
  signOut
} from "https://www.gstatic.com/firebasejs/10.12.4/firebase-auth.js";
import {
  getFirestore,
  collection,
  query,
  where,
  orderBy,
  limit,
  getDocs,
  doc,
  setDoc,
  serverTimestamp
} from "https://www.gstatic.com/firebasejs/10.12.4/firebase-firestore.js";

const page = document.body?.dataset?.page || "";
const $ = (id) => document.getElementById(id);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const firebaseConfigured = Object.values(FIREBASE_CONFIG || {}).every(
  (value) => typeof value === "string" && value.length > 0 && !value.startsWith("YOUR_")
);

let firebaseApp = null;
let auth = null;
let db = null;

if (firebaseConfigured) {
  firebaseApp = initializeApp(FIREBASE_CONFIG);
  auth = getAuth(firebaseApp);
  db = getFirestore(firebaseApp);
}

const toast = $("toast");
const signinBanner = $("signin-banner");
const userChip = $("user-chip");
const userName = $("user-name");
const topLogoutBtn = $("top-logout-btn");
const adminLoginBtn = $("admin-login-btn");
const adminModal = $("admin-modal");
const adminSubmit = $("admin-submit");
const adminCancel = $("admin-cancel");
const adminUserInput = $("admin-username");
const adminPassInput = $("admin-password");
const adminError = $("admin-error");
const adminNavLinks = $$(".admin-only");
const navLinks = $$(".nav-link");
const bottomLinks = $$(".bottom-link");
const brandName = $("brand-name");
const brandSub = $("brand-sub");

const uploadInput = $("upload-input");
const uploadDrop = $("upload-drop");
const cameraInput = $("camera-input");
const cameraBtn = $("camera-btn");
const uploadName = $("upload-name");
const uploadSize = $("upload-size");
const uploadStatus = $("upload-status");
const autoProcess = $("auto-process");
const startBtn = $("start-btn");
const progressBar = $("progress-bar");
const progressLabel = $("progress-label");
const statusFoot = $("status-foot");
const inferenceTime = $("inference-time");
const verdictBadge = $("verdict-badge");
const summaryGrid = $("summary-grid");
const openReviewBtn = $("open-review-btn");
const recentList = $("recent-list");
const previewCard = $("preview-card");
const previewViewer = $("preview-viewer");
const previewImage = $("preview-image");
const previewOverlay = $("preview-overlay");
const previewEmpty = $("preview-empty");
const previewScan = $("preview-scan");

const reviewJobMeta = $("review-job-meta");
const reviewPageMeta = $("review-page-meta");
const reviewImage = $("review-image");
const reviewOverlay = $("review-overlay");
const reviewViewerEmpty = document.querySelector(".viewer-empty");
const prevPageBtn = $("prev-page");
const nextPageBtn = $("next-page");
const regionMeta = $("region-meta");
const reviewNotes = $("review-notes");
const btnReviewed = $("btn-reviewed");
const btnFp = $("btn-fp");
const classLegend = $("class-legend");
const reviewSummary = $("review-summary");
const reviewJobSelect = $("review-job-select");
const reviewLoadBtn = $("review-load-btn");
const reviewJobHint = $("review-job-hint");

const historyList = $("history-list");
const userStats = $("user-stats");
const historyCharts = $("history-charts");
const historyModal = $("history-modal");
const historyModalClose = $("history-modal-close");
const historyModalTitle = $("history-modal-title");
const historyModalMeta = $("history-modal-meta");
const historyModalImage = $("history-modal-image");
const historyModalOverlay = $("history-modal-overlay");
const historyModalEmpty = $("history-modal-empty");
const historyReviewBtn = $("history-review-btn");

const insightGrid = $("insight-grid");

const adminUsers = $("admin-users");
const adminClassSummary = $("admin-class-summary");
const adminKpis = $("admin-kpis");
const adminLocked = $("admin-locked");
const adminDashboard = $("admin-dashboard");

const statUploads = $("stat-uploads");
const statFlags = $("stat-flags");
const statAvg = $("stat-avg");

const FLASH_KEY = "forgensic_flash";
const LAST_JOB_KEY = "forgensic_last_job";
const BANNER_KEY = "forgensic_banner_seen";

let currentUser = null;
let selectedFile = null;
let activeJobId = null;
let resultsPayload = null;
let activePageIndex = 0;
let selectedRegionIndex = null;
const imageObjectUrls = new Map();
let isAdmin = localStorage.getItem("forgensic_admin") === "true";
let jobsCache = null;
let reviewsCache = null;
let resultCache = new Map();
let adminJobsCache = null;
let adminReviewsCache = null;
let pageInitialized = false;
let reviewJobList = [];
let pendingReviewJobId = null;

function setBanner(message) {
  if (!signinBanner) return;
  clearTimeout(signinBanner._hideId);
  signinBanner.textContent = message || "";
  if (!message) {
    signinBanner.classList.remove("visible");
    signinBanner.classList.add("hidden");
    return;
  }
  signinBanner.classList.remove("hidden");
  requestAnimationFrame(() => signinBanner.classList.add("visible"));
  signinBanner._hideId = setTimeout(() => {
    signinBanner.classList.remove("visible");
    signinBanner._hideId = setTimeout(() => {
      signinBanner.classList.add("hidden");
    }, 300);
  }, 3500);
}

function showToast(message, variant = "info") {
  if (!toast) return;
  toast.textContent = message;
  toast.dataset.variant = variant;
  toast.classList.remove("hidden");
  clearTimeout(toast._timeoutId);
  toast._timeoutId = setTimeout(() => {
    toast.classList.add("hidden");
  }, 2800);
}

function redirectToLanding(message) {
  if (message) localStorage.setItem(FLASH_KEY, message);
  window.location.href = "index.html";
}

function updateBrand() {
  $$(".brand-name").forEach((el) => {
    el.textContent = APP_BRAND.name;
  });
  $$(".brand-sub").forEach((el) => {
    el.textContent = APP_BRAND.tagline;
  });
}

function setActiveNav() {
  navLinks.forEach((link) => {
    link.classList.toggle("active", link.dataset.page === page);
  });
  bottomLinks.forEach((link) => {
    link.classList.toggle("active", link.dataset.page === page);
  });
}

function applyAdminState() {
  adminNavLinks.forEach((link) => link.classList.toggle("hidden", !isAdmin));
  if (adminLoginBtn) {
    adminLoginBtn.textContent = isAdmin ? "Admin unlocked" : "Admin login";
  }
  if (adminLocked && adminDashboard) {
    adminLocked.classList.toggle("hidden", isAdmin);
    adminDashboard.classList.toggle("hidden", !isAdmin);
  }
}

function setUserState(user) {
  currentUser = user;
  if (userChip) userChip.classList.toggle("auth-hidden", !user);
  if (topLogoutBtn) topLogoutBtn.classList.toggle("auth-hidden", !user);
  if (adminLoginBtn) adminLoginBtn.classList.toggle("auth-hidden", !user);
  if (userName && user) userName.textContent = user.displayName || user.email || "Customer";

  if (!user) {
    if (firebaseConfigured) {
      redirectToLanding("Please sign in to access Forgensic.");
      return;
    }
    setBanner("Demo mode · Sign-in disabled");
    return;
  }

  const bannerShown = sessionStorage.getItem(BANNER_KEY) === "true";
  if (!bannerShown) {
    setBanner(`Signed in as ${user.displayName || user.email || "User"}`);
    sessionStorage.setItem(BANNER_KEY, "true");
  } else {
    setBanner("");
  }
}

function setStatus(label, progress) {
  if (!progressLabel || !progressBar) return;
  progressLabel.textContent = label;
  progressBar.style.width = `${Math.max(0, Math.min(progress, 1)) * 100}%`;
}

function setScanningState(active) {
  if (!previewCard) return;
  previewCard.classList.toggle("scanning", active);
  if (previewScan) previewScan.classList.toggle("active", active);
  if (previewEmpty) {
    previewEmpty.textContent = active ? "Processing preview..." : "No results yet.";
  }
  if (active) previewCard.classList.remove("hidden");
}

function formatBytes(bytes) {
  if (!bytes) return "0 MB";
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

const IST_FORMATTER = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Asia/Kolkata",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false
});

function formatDateTimeIST(date) {
  const parts = IST_FORMATTER.formatToParts(date);
  const lookup = {};
  parts.forEach((part) => {
    if (part.type !== "literal") {
      lookup[part.type] = part.value;
    }
  });
  return `${lookup.day}-${lookup.month}-${lookup.year}-T-${lookup.hour}:${lookup.minute}:${lookup.second}`;
}

function formatSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  if (value < 1) return `${Math.round(value * 1000)} ms`;
  const precision = value < 10 ? 2 : 1;
  return `${value.toFixed(precision)} s`;
}

function setInferenceTime(totalSeconds, avgSeconds) {
  if (!inferenceTime) return;
  if (totalSeconds === null || totalSeconds === undefined || Number.isNaN(totalSeconds)) {
    inferenceTime.textContent = "";
    return;
  }
  const hasAvg = avgSeconds !== null && avgSeconds !== undefined && !Number.isNaN(avgSeconds);
  const avgLabel = hasAvg ? ` (avg ${formatSeconds(avgSeconds)} / page)` : "";
  inferenceTime.textContent = `Inference time: ${formatSeconds(totalSeconds)}${avgLabel}`;
}

function formatTimestamp(value) {
  if (!value) return "";
  if (typeof value === "string") {
    if (/^\d{2}-\d{2}-\d{4}-T-\d{2}:\d{2}:\d{2}$/.test(value)) {
      return value;
    }
    const parsed = new Date(value);
    if (!Number.isNaN(parsed.getTime())) {
      return formatDateTimeIST(parsed);
    }
    return value;
  }
  if (typeof value.toDate === "function") {
    return formatDateTimeIST(value.toDate());
  }
  if (value instanceof Date) {
    return formatDateTimeIST(value);
  }
  return "";
}

function getJobId(job) {
  return job?.job_id || job?.jobId || job?.result?.job_id || job?.id || "";
}

function matchesJobId(job, jobId) {
  if (!job || !jobId) return false;
  return (
    job.job_id === jobId ||
    job.jobId === jobId ||
    job.result?.job_id === jobId ||
    job.id === jobId
  );
}

function findJobRecord(jobId) {
  const listHit = (reviewJobList || []).find((job) => matchesJobId(job, jobId));
  if (listHit) return listHit;
  const cacheHit = (jobsCache || []).find((job) => matchesJobId(job, jobId));
  return cacheHit || null;
}

function resolveApiJobId(jobId, jobRecord) {
  const record = jobRecord || findJobRecord(jobId);
  if (record?.job_id) return record.job_id;
  if (record?.jobId) return record.jobId;
  if (record?.result?.job_id) return record.result.job_id;
  return jobId;
}

function getInferenceSeconds(job) {
  if (!job) return null;
  if (typeof job.inference_seconds === "number") return job.inference_seconds;
  if (typeof job.inference_ms === "number") return job.inference_ms / 1000;
  if (typeof job.result?.inference_seconds === "number") return job.result.inference_seconds;
  if (typeof job.result?.inference_ms === "number") return job.result.inference_ms / 1000;
  return null;
}

function getAvgInferenceSeconds(job) {
  if (!job) return null;
  if (typeof job.avg_inference_seconds === "number") return job.avg_inference_seconds;
  if (typeof job.result?.avg_inference_seconds === "number") return job.result.avg_inference_seconds;
  return null;
}

function positionOverlayToImage(targetImage, targetOverlay) {
  if (!targetImage || !targetOverlay) return;
  const viewer = targetImage.closest(".viewer");
  if (!viewer) return;
  const viewerRect = viewer.getBoundingClientRect();
  const imageRect = targetImage.getBoundingClientRect();
  if (!imageRect.width || !imageRect.height) return;
  const left = imageRect.left - viewerRect.left;
  const top = imageRect.top - viewerRect.top;
  targetOverlay.style.width = `${imageRect.width}px`;
  targetOverlay.style.height = `${imageRect.height}px`;
  targetOverlay.style.left = `${left}px`;
  targetOverlay.style.top = `${top}px`;
}

function buildJobLabel(job) {
  const name = job?.file_name || "Document";
  const time = formatTimestamp(job?.created_at);
  return time ? `${name} - ${time}` : name;
}

function updateReviewHint(job) {
  if (!reviewJobHint) return;
  reviewJobHint.textContent = job
    ? `Ready to review ${job.file_name || "Document"}.`
    : "Choose a past upload to review.";
}

function populateReviewJobs(jobs) {
  if (!reviewJobSelect || !reviewLoadBtn) return;
  reviewJobList = jobs || [];
  reviewJobSelect.innerHTML = "";

  if (!reviewJobList.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No uploads available";
    reviewJobSelect.appendChild(option);
    reviewJobSelect.disabled = true;
    reviewLoadBtn.disabled = true;
    if (reviewJobHint) reviewJobHint.textContent = "No uploads yet. Process a document first.";
    return;
  }

  reviewJobSelect.disabled = false;
  reviewLoadBtn.disabled = false;

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Choose a job...";
  reviewJobSelect.appendChild(placeholder);

  const targetJobId = activeJobId || pendingReviewJobId;

  reviewJobList.forEach((job) => {
    const jobId = getJobId(job);
    if (!jobId) return;
    const option = document.createElement("option");
    option.value = jobId;
    option.textContent = buildJobLabel(job);
    if (jobId === targetJobId) {
      option.selected = true;
    }
    reviewJobSelect.appendChild(option);
  });

  const selectedJob = reviewJobList.find((job) => getJobId(job) === reviewJobSelect.value);
  updateReviewHint(selectedJob || null);
}

async function hydrateHistoryThumbnail(job, imageEl, placeholderEl) {
  const jobId = getJobId(job);
  if (!jobId || !imageEl) return;
  const apiJobId = resolveApiJobId(jobId, job);
  let pageData = job?.result?.pages?.[0] || null;
  if (!pageData) {
    const payload = await fetchJobResults(apiJobId, job);
    pageData = payload?.pages?.[0] || null;
  }
  if (!pageData?.image_url) return;
  await loadImageForTarget(pageData.image_url, imageEl, placeholderEl);
}

function buildCategorySummary(pages) {
  const counts = {};
  pages.forEach((pageItem) => {
    (pageItem.categories || []).forEach((cat) => {
      counts[cat] = (counts[cat] || 0) + 1;
    });
  });
  return counts;
}

function renderSummary(pages, targetGrid, badge) {
  const summary = buildCategorySummary(pages || []);
  const keys = Object.keys(summary).sort();
  if (targetGrid) {
    targetGrid.innerHTML = "";
    if (!keys.length) {
      targetGrid.innerHTML = "<div class='card-meta'>No detections yet.</div>";
    } else {
      keys.forEach((key) => {
        const tile = document.createElement("div");
        tile.className = "summary-tile";
        tile.innerHTML = `<strong>${key}</strong><div>${summary[key]} pages</div>`;
        targetGrid.appendChild(tile);
      });
    }
  }

  if (badge) {
    badge.textContent = keys.length
      ? keys.includes("C10") && keys.length === 1
        ? "Clean"
        : "Tampering detected"
      : "Awaiting run";
  }
}

function renderLegend() {
  if (!classLegend) return;
  classLegend.innerHTML = "";
  Object.keys(CATEGORY_COLORS).forEach((key) => {
    const item = document.createElement("div");
    item.className = "legend-item";
    item.innerHTML = `
      <span class="legend-swatch" style="background:${CATEGORY_COLORS[key]}"></span>
      <span>${key}</span>
    `;
    classLegend.appendChild(item);
  });
}

function selectFile(file) {
  selectedFile = file;
  if (uploadName) uploadName.textContent = file ? file.name : "No file selected";
  if (uploadSize) uploadSize.textContent = file ? formatBytes(file.size) : "0 MB";
  if (uploadStatus) uploadStatus.textContent = file ? "Ready to process" : "Ready";
  if (file && autoProcess?.checked) {
    startAnalysis();
  }
}

async function loadImageForTarget(imageUrl, targetImage, emptyEl) {
  if (!targetImage) return;
  const viewer = targetImage.closest(".viewer");
  const thumb = targetImage.closest(".history-thumb");
  const resolvedUrl = imageUrl.startsWith("http") ? imageUrl : `${API_BASE_URL}${imageUrl}`;
  const isLocalApi = resolvedUrl.startsWith(API_BASE_URL);
  const token = currentUser ? await currentUser.getIdToken() : null;

  if (!isLocalApi || !token) {
    targetImage.src = resolvedUrl;
  } else {
    try {
      const res = await fetch(resolvedUrl, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (!res.ok) throw new Error("image fetch failed");
      const blob = await res.blob();
      const prevUrl = imageObjectUrls.get(targetImage);
      if (prevUrl) {
        URL.revokeObjectURL(prevUrl);
      }
      const objectUrl = URL.createObjectURL(blob);
      imageObjectUrls.set(targetImage, objectUrl);
      targetImage.src = objectUrl;
    } catch (err) {
      targetImage.src = resolvedUrl;
    }
  }

  targetImage.style.display = "block";
  if (viewer) viewer.classList.add("has-image");
  if (thumb) thumb.classList.add("has-image");
  if (emptyEl) emptyEl.style.display = "none";
}

async function loadReviewImage(imageUrl) {
  return loadImageForTarget(imageUrl, reviewImage, reviewViewerEmpty);
}

async function loadPreviewImage(imageUrl) {
  return loadImageForTarget(imageUrl, previewImage, previewEmpty);
}

function renderStaticOverlay(pageData, targetImage, targetOverlay) {
  if (!pageData || !targetImage || !targetOverlay) return;
  const renderBoxes = () => {
    const imgWidth = pageData.image_width || targetImage.naturalWidth;
    const imgHeight = pageData.image_height || targetImage.naturalHeight;
    if (!imgWidth || !imgHeight) return;
    positionOverlayToImage(targetImage, targetOverlay);
    targetOverlay.innerHTML = "";
    (pageData.regions || []).forEach((region) => {
      const box = document.createElement("div");
      const left = (region.x / imgWidth) * 100;
      const top = (region.y / imgHeight) * 100;
      const width = (region.w / imgWidth) * 100;
      const height = (region.h / imgHeight) * 100;
      const color = CATEGORY_COLORS[region.category_id] || "#f97316";
      box.className = "box";
      box.style.left = `${left}%`;
      box.style.top = `${top}%`;
      box.style.width = `${width}%`;
      box.style.height = `${height}%`;
      box.style.borderColor = color;
      box.style.backgroundColor = `${color}22`;
      targetOverlay.appendChild(box);
    });
  };

  targetImage.onload = () => requestAnimationFrame(renderBoxes);
  if (targetImage.complete) {
    requestAnimationFrame(renderBoxes);
  }
}

function selectRegion(region, index) {
  selectedRegionIndex = index;
  if (!regionMeta) return;
  regionMeta.textContent = region
    ? `${region.category_id} | ${region.type || "flagged"} | ${region.w}x${region.h}`
    : "Select a box to review.";
  document.querySelectorAll(".box").forEach((box) => box.classList.remove("selected"));
  if (index !== null) {
    const target = reviewOverlay?.querySelector(`[data-index='${index}']`);
    if (target) target.classList.add("selected");
  }
}

function renderPage(index) {
  if (!resultsPayload || !resultsPayload.pages?.length || !reviewOverlay) return;
  const pageData = resultsPayload.pages[index];
  if (!pageData) return;

  selectRegion(null, null);

  if (reviewPageMeta) {
    reviewPageMeta.textContent = `Page ${pageData.page_number} | ${(pageData.categories || []).join(", ")}`;
  }

  if (pageData.image_url) {
    loadReviewImage(pageData.image_url);
  }

  const renderBoxes = () => {
    const imgWidth = pageData.image_width || reviewImage?.naturalWidth;
    const imgHeight = pageData.image_height || reviewImage?.naturalHeight;
    if (!imgWidth || !imgHeight) return;
    positionOverlayToImage(reviewImage, reviewOverlay);
    reviewOverlay.innerHTML = "";
    (pageData.regions || []).forEach((region, idx) => {
      const box = document.createElement("div");
      const left = (region.x / imgWidth) * 100;
      const top = (region.y / imgHeight) * 100;
      const width = (region.w / imgWidth) * 100;
      const height = (region.h / imgHeight) * 100;
      const color = CATEGORY_COLORS[region.category_id] || "#f97316";
      box.className = "box";
      box.style.left = `${left}%`;
      box.style.top = `${top}%`;
      box.style.width = `${width}%`;
      box.style.height = `${height}%`;
      box.style.borderColor = color;
      box.style.backgroundColor = `${color}22`;
      box.dataset.index = idx;
      reviewOverlay.appendChild(box);
    });
  };

  if (reviewImage) {
    reviewImage.onload = () => requestAnimationFrame(renderBoxes);
    if (reviewImage.complete) {
      requestAnimationFrame(renderBoxes);
    }
  }
}

function renderDashboardPreview(pageData) {
  if (!previewCard || !previewOverlay || !previewImage || !pageData) return;
  previewCard.classList.remove("hidden");

  if (pageData.image_url) {
    loadPreviewImage(pageData.image_url);
  }
  renderStaticOverlay(pageData, previewImage, previewOverlay);
}

async function saveReview(status) {
  if (!db || !currentUser || !resultsPayload) return;
  const pageData = resultsPayload.pages?.[activePageIndex];
  if (!pageData || selectedRegionIndex === null) return;

  const reviewId = `${resultsPayload.job_id}_${pageData.page_number}_${selectedRegionIndex}_${currentUser.uid}`;
  await setDoc(
    doc(db, "reviews", reviewId),
    {
      job_id: resultsPayload.job_id,
      page_number: pageData.page_number,
      region_index: selectedRegionIndex,
      status,
      note: reviewNotes?.value || "",
      user_id: currentUser.uid,
      user_email: currentUser.email || "",
      updated_at: serverTimestamp(),
      created_at: serverTimestamp()
    },
    { merge: true }
  );
  reviewsCache = null;
  showToast("Review saved", "info");
}

async function pollJob(jobId) {
  setStatus("Queued", 0.1);
  if (statusFoot) statusFoot.textContent = "Job queued";

  const interval = setInterval(async () => {
    const token = currentUser ? await currentUser.getIdToken() : null;
    const res = await fetch(`${API_BASE_URL}/jobs/${jobId}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    });
    if (!res.ok) return;
    const data = await res.json();
    setStatus(data.status, data.progress || 0.2);
    if (statusFoot) statusFoot.textContent = data.status;

    if (data.status === "queued" || data.status === "processing") {
      setScanningState(true);
    }

    if (data.status === "complete") {
      clearInterval(interval);
      setScanningState(false);
      await loadResults(jobId);
    }
    if (data.status === "error") {
      clearInterval(interval);
      setScanningState(false);
      if (statusFoot) statusFoot.textContent = data.message || "Pipeline error";
    }
  }, 2000);
}

async function loadResults(jobId) {
  const jobRecord = findJobRecord(jobId);
  const apiJobId = resolveApiJobId(jobId, jobRecord);
  activeJobId = apiJobId;
  pendingReviewJobId = apiJobId;
  if (reviewJobSelect) reviewJobSelect.value = apiJobId;

  const payload = await fetchJobResults(apiJobId, jobRecord);
  if (!payload) {
    if (!jobRecord && !jobsCache && page === "review") {
      if (reviewSummary) reviewSummary.innerHTML = "<div class='card-meta'>Waiting for job details...</div>";
      return;
    }
    if (statusFoot) statusFoot.textContent = "Failed to load results";
    if (reviewSummary) reviewSummary.innerHTML = "<div class='card-meta'>No results found for this job.</div>";
    showToast("No results found for this job", "error");
    return;
  }

  resultsPayload = payload;
  if (!resultsPayload.pages || !resultsPayload.pages.length) {
    if (statusFoot) statusFoot.textContent = "No pages detected";
    if (reviewSummary) reviewSummary.innerHTML = "<div class='card-meta'>No pages detected for this job.</div>";
    return;
  }

  const totalInference = getInferenceSeconds(resultsPayload);
  const avgInference =
    resultsPayload.avg_inference_seconds ||
    (totalInference && resultsPayload.pages?.length
      ? totalInference / resultsPayload.pages.length
      : null);
  setInferenceTime(totalInference, avgInference);

  setScanningState(false);
  resultCache.set(apiJobId, resultsPayload);

  localStorage.setItem(LAST_JOB_KEY, apiJobId);
  if (openReviewBtn) {
    openReviewBtn.href = `review.html?job=${apiJobId}`;
  }

  renderSummary(resultsPayload.pages, summaryGrid, verdictBadge);
  renderSummary(resultsPayload.pages, reviewSummary, null);
  renderLegend();
  renderDashboardPreview(resultsPayload.pages[0]);

  activePageIndex = 0;
  renderPage(activePageIndex);
  const firstRegion = resultsPayload.pages[0].regions?.[0] || null;
  selectRegion(firstRegion, firstRegion ? 0 : null);

  if (statusFoot) statusFoot.textContent = "Complete";
  if (reviewJobMeta) reviewJobMeta.textContent = `Job ${apiJobId}`;

  jobsCache = null;
  reviewsCache = null;
  await loadUserData();
  if (isAdmin) await loadAdmin();
}

async function fetchJobResults(jobId, jobRecord = null) {
  if (!jobId) return null;
  const apiJobId = resolveApiJobId(jobId, jobRecord);
  if (resultCache.has(apiJobId)) return resultCache.get(apiJobId);
  if (jobRecord?.result?.pages?.length) {
    const payload = { ...jobRecord.result, job_id: apiJobId };
    resultCache.set(apiJobId, payload);
    return payload;
  }
  const token = currentUser ? await currentUser.getIdToken() : null;
  const res = await fetch(`${API_BASE_URL}/jobs/${apiJobId}/results`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {}
  });
  if (!res.ok) return null;
  const payload = await res.json();
  payload.job_id = payload.job_id || apiJobId;
  resultCache.set(apiJobId, payload);
  return payload;
}

async function startAnalysis() {
  if (!currentUser) {
    if (uploadStatus) uploadStatus.textContent = "Please sign in first";
    if (auth) {
      redirectToLanding("Please sign in first.");
    } else {
      showToast("Demo mode: sign-in is disabled.", "info");
    }
    return;
  }
  if (!selectedFile) {
    if (uploadStatus) uploadStatus.textContent = "Please select a file";
    return;
  }
  if (selectedFile.size > APP_OPTIONS.maxUploadBytes) {
    if (uploadStatus) uploadStatus.textContent = "File exceeds 10 MB limit";
    return;
  }

  setInferenceTime(null, null);

  const token = currentUser ? await currentUser.getIdToken() : null;
  const formData = new FormData();
  formData.append("file", selectedFile);

  const res = await fetch(`${API_BASE_URL}/jobs`, {
    method: "POST",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData
  });

  if (!res.ok) {
    if (uploadStatus) uploadStatus.textContent = "Upload failed";
    return;
  }
  const data = await res.json();
  activeJobId = data.job_id;
  if (uploadStatus) uploadStatus.textContent = "Processing started";
  localStorage.setItem(LAST_JOB_KEY, activeJobId);
  setScanningState(true);
  pollJob(activeJobId);
}

async function fetchUserJobs() {
  if (!currentUser || !db) return [];
  if (jobsCache) return jobsCache;
  let snapshot;
  try {
    const q = query(
      collection(db, "jobs"),
      where("user_id", "==", currentUser.uid),
      orderBy("created_at", "desc"),
      limit(40)
    );
    snapshot = await getDocs(q);
  } catch (err) {
    const message = err?.message || "";
    if (err?.code === "failed-precondition" || message.includes("index")) {
      const fallback = query(
        collection(db, "jobs"),
        where("user_id", "==", currentUser.uid),
        limit(40)
      );
      snapshot = await getDocs(fallback);
    } else {
      throw err;
    }
  }
  jobsCache = snapshot.docs.map((docSnap) => ({ id: docSnap.id, ...docSnap.data() }));
  return jobsCache;
}

async function fetchUserReviews() {
  if (!currentUser || !db) return [];
  if (reviewsCache) return reviewsCache;
  const q = query(collection(db, "reviews"), where("user_id", "==", currentUser.uid), limit(200));
  const snapshot = await getDocs(q);
  reviewsCache = snapshot.docs.map((docSnap) => ({ id: docSnap.id, ...docSnap.data() }));
  return reviewsCache;
}

function buildUserSummary(jobs, reviews) {
  const classCounts = {};
  let totalFlags = 0;
  let totalInferenceSeconds = 0;
  let inferenceJobs = 0;
  jobs.forEach((job) => {
    let summary = job.result?.category_summary;
    if (!summary && job.result?.pages) {
      summary = buildCategorySummary(job.result.pages);
    }
    summary = summary || {};
    Object.keys(summary).forEach((key) => {
      classCounts[key] = (classCounts[key] || 0) + summary[key];
      totalFlags += summary[key];
    });

    const inferenceSeconds = getInferenceSeconds(job);
    if (typeof inferenceSeconds === "number") {
      totalInferenceSeconds += inferenceSeconds;
      inferenceJobs += 1;
    }
  });
  const reviewCounts = { reviewed: 0, false_positive: 0 };
  reviews.forEach((review) => {
    if (review.status === "reviewed") reviewCounts.reviewed += 1;
    if (review.status === "false_positive") reviewCounts.false_positive += 1;
  });
  return {
    totalUploads: jobs.length,
    totalFlags,
    classCounts,
    reviewCounts,
    avgInferenceSeconds: inferenceJobs ? totalInferenceSeconds / inferenceJobs : null
  };
}

function renderHistoryList(items) {
  if (!historyList) return;
  historyList.innerHTML = "";
  if (!items.length) {
    historyList.innerHTML = "<div class='card-meta'>No uploads yet.</div>";
    return;
  }
  items.forEach((data) => {
    const row = document.createElement("div");
    const jobId = getJobId(data);
    row.className = "history-item";
    row.dataset.jobId = jobId || "";
    row.innerHTML = `
      <div class="history-thumb">
        <div class="doc-silhouette" aria-hidden="true">
          <span class="doc-corner" aria-hidden="true"></span>
        </div>
        <img class="history-thumb-image" alt="History preview" />
      </div>
      <div class="history-main">
        <strong>${data.file_name || "Document"}</strong>
        <div class="card-meta">${data.status || "unknown"}</div>
      </div>
      <div class="history-actions">
        <span class="card-meta">${formatTimestamp(data.created_at)}</span>
        <button class="btn btn-ghost btn-mini" type="button">Preview</button>
      </div>
    `;
    const thumbImage = row.querySelector(".history-thumb-image");
    const thumbPlaceholder = row.querySelector(".history-thumb .doc-silhouette");
    if (thumbImage) {
      hydrateHistoryThumbnail(data, thumbImage, thumbPlaceholder);
    }
    row.querySelector("button")?.addEventListener("click", (event) => {
      event.stopPropagation();
      openHistoryModal(data);
    });
    row.addEventListener("click", () => openHistoryModal(data));
    historyList.appendChild(row);
  });
}

async function openHistoryModal(job) {
  if (!historyModal) return;
  const jobId = getJobId(job);
  if (!jobId) return;
  const apiJobId = resolveApiJobId(jobId, job);

  historyModal.classList.remove("hidden");
  if (historyModalTitle) historyModalTitle.textContent = job.file_name || "Document";
  if (historyModalMeta) historyModalMeta.textContent = `${job.status || "unknown"} · ${formatTimestamp(job.created_at)}`;
  if (historyReviewBtn) {
    historyReviewBtn.href = `review.html?job=${apiJobId}`;
    historyReviewBtn.dataset.jobId = apiJobId;
  }
  localStorage.setItem(LAST_JOB_KEY, apiJobId);
  if (historyModalEmpty) {
    historyModalEmpty.textContent = "Loading preview...";
    historyModalEmpty.style.display = "block";
  }
  if (historyModalOverlay) historyModalOverlay.innerHTML = "";

  const payload = job.result?.pages
    ? { ...job.result, job_id: apiJobId }
    : await fetchJobResults(apiJobId, job);
  if (!payload || !payload.pages?.length) {
    if (historyModalEmpty) historyModalEmpty.textContent = "No preview available.";
    return;
  }

  const pageData = payload.pages[0];
  if (pageData?.image_url) {
    await loadImageForTarget(pageData.image_url, historyModalImage, historyModalEmpty);
    renderStaticOverlay(pageData, historyModalImage, historyModalOverlay);
  } else if (historyModalEmpty) {
    historyModalEmpty.textContent = "No preview available.";
  }
}

function closeHistoryModal() {
  if (!historyModal) return;
  historyModal.classList.add("hidden");
}

function renderRecentList(items) {
  if (!recentList) return;
  recentList.innerHTML = "";
  const topItems = items.slice(0, 4);
  if (!topItems.length) {
    recentList.innerHTML = "<div class='card-meta'>No recent uploads yet.</div>";
    return;
  }
  topItems.forEach((data) => {
    const row = document.createElement("div");
    row.className = "recent-item";
    row.innerHTML = `
      <div>
        <strong>${data.file_name || "Document"}</strong>
        <div class="card-meta">${data.status || "unknown"}</div>
      </div>
      <div class="card-meta">${formatTimestamp(data.created_at)}</div>
    `;
    recentList.appendChild(row);
  });
}

function renderStats(summary) {
  if (statUploads) statUploads.textContent = summary.totalUploads.toString();
  if (statFlags) statFlags.textContent = summary.totalFlags.toString();
  if (statAvg) {
    statAvg.textContent = summary.avgInferenceSeconds
      ? formatSeconds(summary.avgInferenceSeconds)
      : "--";
  }
}

function syncAutoProcessState() {
  if (!autoProcess || !startBtn) return;
  const isAuto = autoProcess.checked;
  startBtn.disabled = isAuto;
}

function renderUserStats(summary) {
  if (!userStats) return;
  userStats.innerHTML = "";
  const blocks = [
    { label: "Uploads", value: summary.totalUploads },
    { label: "Flags", value: summary.totalFlags },
    { label: "Reviewed", value: summary.reviewCounts.reviewed },
    { label: "False positives", value: summary.reviewCounts.false_positive }
  ];
  blocks.forEach((item) => {
    const card = document.createElement("div");
    card.className = "summary-tile";
    card.innerHTML = `<strong>${item.label}</strong><div>${item.value}</div>`;
    userStats.appendChild(card);
  });
  Object.keys(summary.classCounts).forEach((key) => {
    const card = document.createElement("div");
    card.className = "summary-tile";
    card.innerHTML = `<strong>${key}</strong><div>${summary.classCounts[key]} flags</div>`;
    userStats.appendChild(card);
  });
}

function renderHistoryCharts(summary) {
  if (!historyCharts) return;
  historyCharts.innerHTML = "";
  const entries = Object.entries(summary.classCounts).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    historyCharts.innerHTML = "<div class='card-meta'>No class trends yet.</div>";
    return;
  }
  const max = Math.max(...entries.map(([, value]) => value));
  entries.forEach(([key, value]) => {
    const row = document.createElement("div");
    row.className = "chart-row";
    row.innerHTML = `
      <span>${key}</span>
      <div class="chart-bar"><div class="chart-fill" style="width:${(value / max) * 100}%"></div></div>
      <span>${value}</span>
    `;
    historyCharts.appendChild(row);
  });
}

async function loadUserData() {
  if (!currentUser || !db) return;
  const jobs = await fetchUserJobs();
  const reviews = await fetchUserReviews();
  const summary = buildUserSummary(jobs, reviews);
  renderStats(summary);
  renderHistoryList(jobs);
  renderRecentList(jobs);
  renderUserStats(summary);
  renderHistoryCharts(summary);
  renderInsights(summary);
  if (page === "review") {
    populateReviewJobs(jobs);
    if (!resultsPayload && pendingReviewJobId) {
      loadResults(pendingReviewJobId);
    }
  }
}

function renderInsights(summary) {
  if (!insightGrid) return;
  insightGrid.innerHTML = "";
  const classes = Object.entries(summary.classCounts).sort((a, b) => b[1] - a[1]);
  const topClasses = classes.slice(0, 5);

  const overview = document.createElement("div");
  overview.className = "card";
  overview.innerHTML = `
    <div class="card-title">Overall activity</div>
    <div class="card-meta">Uploads and reviews across your history.</div>
    <div class="stats-grid">
      <div class="summary-tile"><strong>Uploads</strong><div>${summary.totalUploads}</div></div>
      <div class="summary-tile"><strong>Reviewed</strong><div>${summary.reviewCounts.reviewed}</div></div>
      <div class="summary-tile"><strong>False positives</strong><div>${summary.reviewCounts.false_positive}</div></div>
      <div class="summary-tile"><strong>Flags</strong><div>${summary.totalFlags}</div></div>
      <div class="summary-tile"><strong>Avg inference</strong><div>${formatSeconds(summary.avgInferenceSeconds)}</div></div>
    </div>
  `;
  insightGrid.appendChild(overview);

  const classCard = document.createElement("div");
  classCard.className = "card";
  classCard.innerHTML = `<div class="card-title">Top classes</div><div class="chart"></div>`;
  const chart = classCard.querySelector(".chart");
  if (topClasses.length) {
    const max = Math.max(...topClasses.map(([, value]) => value));
    topClasses.forEach(([key, value]) => {
      const row = document.createElement("div");
      row.className = "chart-row";
      row.innerHTML = `
        <span>${key}</span>
        <div class="chart-bar"><div class="chart-fill" style="width:${(value / max) * 100}%"></div></div>
        <span>${value}</span>
      `;
      chart.appendChild(row);
    });
  } else {
    chart.innerHTML = "<div class='card-meta'>No class data yet.</div>";
  }
  insightGrid.appendChild(classCard);

  const noteCard = document.createElement("div");
  noteCard.className = "card";
  noteCard.innerHTML = `
    <div class="card-title">Review outcomes</div>
    <div class="card-meta">Keep an eye on false positives and review throughput.</div>
    <div class="stats-grid">
      <div class="summary-tile"><strong>Reviewed</strong><div>${summary.reviewCounts.reviewed}</div></div>
      <div class="summary-tile"><strong>False positives</strong><div>${summary.reviewCounts.false_positive}</div></div>
    </div>
  `;
  insightGrid.appendChild(noteCard);
}

async function fetchAdminJobs() {
  if (!db) return [];
  if (adminJobsCache) return adminJobsCache;
  let snapshot;
  try {
    const q = query(collection(db, "jobs"), orderBy("created_at", "desc"), limit(80));
    snapshot = await getDocs(q);
  } catch (err) {
    const message = err?.message || "";
    if (err?.code === "failed-precondition" || message.includes("index")) {
      const fallback = query(collection(db, "jobs"), limit(80));
      snapshot = await getDocs(fallback);
    } else {
      throw err;
    }
  }
  adminJobsCache = snapshot.docs.map((docSnap) => ({ id: docSnap.id, ...docSnap.data() }));
  return adminJobsCache;
}

async function fetchAdminReviews() {
  if (!db) return [];
  if (adminReviewsCache) return adminReviewsCache;
  const q = query(collection(db, "reviews"), limit(200));
  const snapshot = await getDocs(q);
  adminReviewsCache = snapshot.docs.map((docSnap) => ({ id: docSnap.id, ...docSnap.data() }));
  return adminReviewsCache;
}

async function loadAdmin() {
  if (!isAdmin || !db) return;
  if (adminUsers) adminUsers.innerHTML = "";
  if (adminClassSummary) adminClassSummary.innerHTML = "";
  if (adminKpis) adminKpis.innerHTML = "";

  const jobs = await fetchAdminJobs();
  const reviews = await fetchAdminReviews();

  const userMap = {};
  const classTotals = {};
  let totalInferenceSeconds = 0;
  let inferenceJobs = 0;
  jobs.forEach((job) => {
    const userId = job.user_id || "unknown";
    if (!userMap[userId]) {
      userMap[userId] = {
        email: job.user_email || "unknown",
        name: job.user_name || job.user_email || "User",
        uploads: 0,
        last: job.created_at || "",
        reviews: 0,
        falsePositives: 0
      };
    }
    userMap[userId].uploads += 1;
    const summary = job.result?.category_summary || {};
    Object.keys(summary).forEach((key) => {
      classTotals[key] = (classTotals[key] || 0) + summary[key];
    });

    const inferenceSeconds = getInferenceSeconds(job);
    if (typeof inferenceSeconds === "number") {
      totalInferenceSeconds += inferenceSeconds;
      inferenceJobs += 1;
    }
  });

  reviews.forEach((review) => {
    const userId = review.user_id || "unknown";
    if (!userMap[userId]) {
      userMap[userId] = {
        email: review.user_email || "unknown",
        name: review.user_email || "User",
        uploads: 0,
        last: "",
        reviews: 0,
        falsePositives: 0
      };
    }
    userMap[userId].reviews += 1;
    if (review.status === "false_positive") userMap[userId].falsePositives += 1;
  });

  const totalUsers = Object.keys(userMap).length;
  const totalUploads = jobs.length;
  const totalReviews = reviews.length;
  const totalFalsePositives = reviews.filter((review) => review.status === "false_positive").length;
  const avgInferenceSeconds = inferenceJobs ? totalInferenceSeconds / inferenceJobs : null;

  const kpiBlocks = [
    { label: "Total users", value: totalUsers },
    { label: "Total uploads", value: totalUploads },
    { label: "Reviews logged", value: totalReviews },
    { label: "False positives", value: totalFalsePositives },
    { label: "Avg inference", value: formatSeconds(avgInferenceSeconds) }
  ];
  if (adminKpis) {
    kpiBlocks.forEach((item) => {
      const card = document.createElement("div");
      card.className = "summary-tile";
      card.innerHTML = `<strong>${item.label}</strong><div>${item.value}</div>`;
      adminKpis.appendChild(card);
    });
  }

  if (adminClassSummary) {
    Object.keys(classTotals).forEach((key) => {
      const item = document.createElement("div");
      item.className = "legend-item";
      item.innerHTML = `
        <span class="legend-swatch" style="background:${CATEGORY_COLORS[key] || "#f97316"}"></span>
        <span>${key}: ${classTotals[key]}</span>
      `;
      adminClassSummary.appendChild(item);
    });
  }

  if (adminUsers) {
    Object.values(userMap).forEach((user) => {
      const row = document.createElement("div");
      row.className = "admin-row";
      row.innerHTML = `
        <div>
          <strong>${user.name}</strong>
          <div class="card-meta">${user.email}</div>
          <div class="card-meta">Last: ${formatTimestamp(user.last)}</div>
        </div>
        <div>
          <strong>${user.uploads}</strong>
          <div class="card-meta">Uploads</div>
        </div>
        <div>
          <strong>${user.reviews}</strong>
          <div class="card-meta">Reviews</div>
        </div>
        <div>
          <strong>${user.falsePositives}</strong>
          <div class="card-meta">False positives</div>
        </div>
      `;
      adminUsers.appendChild(row);
    });
  }
}

function initDashboard() {
  if (!uploadInput || !uploadDrop) return;

  uploadInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) selectFile(file);
  });

  cameraInput?.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) selectFile(file);
  });

  cameraBtn?.addEventListener("click", () => {
    cameraInput?.click();
  });

  uploadDrop.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadDrop.classList.add("dragover");
  });

  uploadDrop.addEventListener("dragleave", () => uploadDrop.classList.remove("dragover"));

  uploadDrop.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadDrop.classList.remove("dragover");
    const file = event.dataTransfer.files?.[0];
    if (file) selectFile(file);
  });

  startBtn?.addEventListener("click", () => startAnalysis());

  autoProcess?.addEventListener("change", () => {
    syncAutoProcessState();
    if (autoProcess.checked && selectedFile) {
      startAnalysis();
    }
  });

  const lastJobId = localStorage.getItem(LAST_JOB_KEY);
  if (openReviewBtn && (activeJobId || lastJobId)) {
    openReviewBtn.href = `review.html?job=${activeJobId || lastJobId}`;
  }

  setStatus("Idle", 0);
  syncAutoProcessState();
}

function initReview() {
  renderLegend();
  reviewOverlay?.addEventListener("click", (event) => {
    const target = event.target;
    if (!target.classList.contains("box")) return;
    const idx = parseInt(target.dataset.index, 10);
    const pageData = resultsPayload?.pages?.[activePageIndex];
    if (!pageData) return;
    selectRegion(pageData.regions[idx], idx);
  });

  prevPageBtn?.addEventListener("click", () => {
    if (!resultsPayload) return;
    activePageIndex = Math.max(0, activePageIndex - 1);
    renderPage(activePageIndex);
  });

  nextPageBtn?.addEventListener("click", () => {
    if (!resultsPayload) return;
    activePageIndex = Math.min(resultsPayload.pages.length - 1, activePageIndex + 1);
    renderPage(activePageIndex);
  });

  btnReviewed?.addEventListener("click", async () => {
    await saveReview("reviewed");
  });

  btnFp?.addEventListener("click", async () => {
    await saveReview("false_positive");
  });

  reviewJobSelect?.addEventListener("change", () => {
    const jobId = reviewJobSelect.value;
    const selectedJob = reviewJobList.find((job) => getJobId(job) === jobId);
    updateReviewHint(selectedJob || null);
  });

  reviewLoadBtn?.addEventListener("click", () => {
    const jobId = reviewJobSelect?.value || "";
    if (!jobId) {
      showToast("Select a job to review", "error");
      return;
    }
    pendingReviewJobId = jobId;
    loadResults(jobId);
  });

  const params = new URLSearchParams(window.location.search);
  const jobId = params.get("job") || localStorage.getItem(LAST_JOB_KEY);
  pendingReviewJobId = jobId || null;
  if (jobId) {
    loadResults(jobId);
  } else if (reviewSummary) {
    reviewSummary.innerHTML = "<div class='card-meta'>No job selected yet.</div>";
  }
}

function initHistory() {
  historyModalClose?.addEventListener("click", () => closeHistoryModal());
  historyModal?.addEventListener("click", (event) => {
    if (event.target === historyModal) closeHistoryModal();
  });
  historyReviewBtn?.addEventListener("click", () => {
    const jobId = historyReviewBtn.dataset.jobId || "";
    if (jobId) localStorage.setItem(LAST_JOB_KEY, jobId);
  });
}

function initInsights() {
  // Data will be loaded after auth state is ready.
}

function initAdmin() {
  applyAdminState();
}

function initPage() {
  if (pageInitialized) return;
  pageInitialized = true;
  if (page === "dashboard") initDashboard();
  if (page === "review") initReview();
  if (page === "history") initHistory();
  if (page === "insights") initInsights();
  if (page === "admin") initAdmin();
}

function initCommon() {
  updateBrand();
  setActiveNav();
  applyAdminState();

  topLogoutBtn?.addEventListener("click", async () => {
    if (!auth) return;
    localStorage.removeItem("forgensic_admin");
    await signOut(auth);
  });

  adminLoginBtn?.addEventListener("click", () => {
    adminModal?.classList.remove("hidden");
    if (adminError) adminError.textContent = "";
  });

  adminCancel?.addEventListener("click", () => {
    adminModal?.classList.add("hidden");
  });

  adminSubmit?.addEventListener("click", () => {
    const username = adminUserInput?.value.trim() || "";
    const password = adminPassInput?.value.trim() || "";
    if (username === ADMIN_CREDENTIALS.username && password === ADMIN_CREDENTIALS.password) {
      isAdmin = true;
      localStorage.setItem("forgensic_admin", "true");
      adminModal?.classList.add("hidden");
      applyAdminState();
      if (page === "admin") loadAdmin();
      showToast("Admin access unlocked", "info");
      return;
    }
    if (adminError) adminError.textContent = "Invalid credentials";
  });
}

initCommon();

if (auth) {
  onAuthStateChanged(auth, async (user) => {
    setUserState(user);
    if (!user && firebaseConfigured) return;
    initPage();
    await loadUserData();
    if (page === "admin" && isAdmin) {
      await loadAdmin();
    }
  });
} else {
  setUserState(null);
  initPage();
}
