import {
  APP_BRAND,
  QUESTIONNAIRE_KEY,
  QUESTIONNAIRE_REQUIRED,
  CONSENT_KEY,
  CONSENT_REQUIRED
} from "./config.js";

const brandName = document.getElementById("brand-name");
const brandSub = document.getElementById("brand-sub");
const form = document.getElementById("questionnaire-form");
const consentCheck = document.getElementById("consent-check");
const submitBtn = document.getElementById("questionnaire-submit");
const statusEl = document.getElementById("questionnaire-status");

const REQUIRED_SELECTOR = "[data-required]";
const PAYLOAD_KEY = "forgensic_questionnaire_payload";

const consentRequired = CONSENT_REQUIRED !== false;
const consentDone = localStorage.getItem(CONSENT_KEY) === "true";

if (consentRequired && !consentDone) {
  window.location.href = "consent.html";
}

if (brandName) brandName.textContent = APP_BRAND.name;
if (brandSub) brandSub.textContent = APP_BRAND.tagline;

function isComplete() {
  if (!form) return false;
  const requiredFields = Array.from(form.querySelectorAll(REQUIRED_SELECTOR));
  return requiredFields.every((field) => {
    if (field instanceof HTMLInputElement || field instanceof HTMLSelectElement || field instanceof HTMLTextAreaElement) {
      return field.value.trim().length > 0;
    }
    return true;
  });
}

function updateSubmitState() {
  if (!submitBtn || !consentCheck) return;
  if (QUESTIONNAIRE_REQUIRED === false) {
    submitBtn.disabled = false;
    return;
  }
  submitBtn.disabled = !(consentCheck.checked && isComplete());
}

function setStatus(message) {
  if (!statusEl) return;
  statusEl.textContent = message || "";
}

form?.addEventListener("input", updateSubmitState);
form?.addEventListener("submit", (event) => {
  event.preventDefault();
  if (QUESTIONNAIRE_REQUIRED !== false && (!consentCheck?.checked || !isComplete())) {
    setStatus("Please complete all required fields and confirm consent.");
    return;
  }
  const payload = Object.fromEntries(new FormData(form).entries());
  payload.consent = consentCheck?.checked ? "true" : "false";
  payload.submitted_at = new Date().toISOString();
  localStorage.setItem(PAYLOAD_KEY, JSON.stringify(payload));
  localStorage.setItem(QUESTIONNAIRE_KEY, "true");
  window.location.href = "app.html";
});

const alreadyDone = localStorage.getItem(QUESTIONNAIRE_KEY) === "true";
if (alreadyDone) {
  setStatus("Questionnaire already completed. You can update answers or continue.");
}

updateSubmitState();
