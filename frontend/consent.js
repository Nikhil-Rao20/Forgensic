import {
  APP_BRAND,
  CONSENT_KEY,
  CONSENT_REQUIRED,
  QUESTIONNAIRE_KEY,
  QUESTIONNAIRE_REQUIRED
} from "./config.js";

const brandName = document.getElementById("brand-name");
const brandSub = document.getElementById("brand-sub");
const form = document.getElementById("consent-form");
const consentCheck = document.getElementById("consent-check");
const acceptBtn = document.getElementById("consent-accept");
const statusEl = document.getElementById("consent-status");

if (brandName) brandName.textContent = APP_BRAND.name;
if (brandSub) brandSub.textContent = APP_BRAND.tagline;

function isQuestionnaireComplete() {
  if (QUESTIONNAIRE_REQUIRED === false) return true;
  return localStorage.getItem(QUESTIONNAIRE_KEY) === "true";
}

function updateButtonState() {
  if (!acceptBtn || !consentCheck) return;
  if (CONSENT_REQUIRED === false) {
    acceptBtn.disabled = false;
    return;
  }
  acceptBtn.disabled = !consentCheck.checked;
}

function setStatus(message) {
  if (!statusEl) return;
  statusEl.textContent = message || "";
}

form?.addEventListener("input", updateButtonState);
form?.addEventListener("submit", (event) => {
  event.preventDefault();
  if (CONSENT_REQUIRED !== false && !consentCheck?.checked) {
    setStatus("Please confirm the consent checkbox to continue.");
    return;
  }
  localStorage.setItem(CONSENT_KEY, "true");
  localStorage.setItem("forgensic_consent_timestamp", new Date().toISOString());
  if (!isQuestionnaireComplete()) {
    window.location.href = "questionnaire.html";
    return;
  }
  window.location.href = "app.html";
});

const alreadyDone = localStorage.getItem(CONSENT_KEY) === "true";
if (alreadyDone) {
  setStatus("Consent already accepted. You can continue to the dashboard.");
}

updateButtonState();
