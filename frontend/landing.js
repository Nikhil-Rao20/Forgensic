import {
  FIREBASE_CONFIG,
  APP_BRAND,
  QUESTIONNAIRE_KEY,
  QUESTIONNAIRE_REQUIRED,
  CONSENT_KEY,
  CONSENT_REQUIRED
} from "./config.js";
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.4/firebase-app.js";
import {
  getAuth,
  signInWithPopup,
  signInWithRedirect,
  GoogleAuthProvider,
  getRedirectResult,
  onAuthStateChanged
} from "https://www.gstatic.com/firebasejs/10.12.4/firebase-auth.js";

const firebaseConfigured = Object.values(FIREBASE_CONFIG || {}).every(
  (value) => typeof value === "string" && value.length > 0 && !value.startsWith("YOUR_")
);

let auth = null;
if (firebaseConfigured) {
  const app = initializeApp(FIREBASE_CONFIG);
  auth = getAuth(app);
}

const landingLoginBtn = document.getElementById("landing-login-btn");
const landingStartBtn = document.getElementById("landing-start-btn");
const landingError = document.getElementById("landing-error");
const brandName = document.getElementById("brand-name");
const brandSub = document.getElementById("brand-sub");
const flashMessage = localStorage.getItem("forgensic_flash");

let authBusy = false;

function setLandingError(message) {
  if (!landingError) return;
  landingError.textContent = message || "";
  landingError.classList.toggle("hidden", !message);
}

function setAuthButtonsEnabled(enabled) {
  if (landingLoginBtn) landingLoginBtn.disabled = !enabled;
  if (landingStartBtn) landingStartBtn.disabled = !enabled;
}

function redirectToApp() {
  window.location.href = "app.html";
}

function redirectToQuestionnaire() {
  window.location.href = "questionnaire.html";
}

function redirectToConsent() {
  window.location.href = "consent.html";
}

function isConsentComplete() {
  if (CONSENT_REQUIRED === false) return true;
  return localStorage.getItem(CONSENT_KEY) === "true";
}

function isQuestionnaireComplete() {
  if (QUESTIONNAIRE_REQUIRED === false) return true;
  return localStorage.getItem(QUESTIONNAIRE_KEY) === "true";
}

function getNextStep() {
  if (!isConsentComplete()) return "consent";
  if (!isQuestionnaireComplete()) return "questionnaire";
  return "app";
}

function updateCtas() {
  const next = getNextStep();
  if (landingLoginBtn) {
    landingLoginBtn.textContent = next === "app" ? "Continue to console" : "Accept e-consent";
  }
  if (landingStartBtn) {
    landingStartBtn.textContent = next === "questionnaire" ? "Start questionnaire" : "View consent";
  }
}

async function signInWithGoogle() {
  if (!auth || authBusy) {
    const next = getNextStep();
    if (next === "consent") redirectToConsent();
    if (next === "questionnaire") redirectToQuestionnaire();
    if (next === "app") redirectToApp();
    return;
  }
  authBusy = true;
  setLandingError("");
  setAuthButtonsEnabled(false);
  const provider = new GoogleAuthProvider();
  try {
    await signInWithPopup(auth, provider);
    const next = getNextStep();
    if (next === "consent") redirectToConsent();
    if (next === "questionnaire") redirectToQuestionnaire();
    if (next === "app") redirectToApp();
  } catch (err) {
    const code = err?.code || "";
    if (code === "auth/cancelled-popup-request") {
      setLandingError("Sign-in already open. Please finish the popup.");
      return;
    }
    if (code === "auth/popup-blocked" || code === "auth/popup-closed-by-user") {
      setLandingError("Popup blocked or closed. Redirecting to sign-in...");
      await signInWithRedirect(auth, provider);
      return;
    }
    if (code === "auth/unauthorized-domain") {
      setLandingError("This URL is not authorized in Firebase Auth.");
      return;
    }
    setLandingError(`Sign-in failed (${code || "unknown"}).`);
  } finally {
    authBusy = false;
    setAuthButtonsEnabled(true);
  }
}

landingLoginBtn?.addEventListener("click", () => signInWithGoogle());
landingStartBtn?.addEventListener("click", () => {
  const next = getNextStep();
  if (next === "consent") redirectToConsent();
  if (next === "questionnaire") redirectToQuestionnaire();
  if (next === "app") redirectToApp();
});

if (brandName) brandName.textContent = APP_BRAND.name;
if (brandSub) brandSub.textContent = APP_BRAND.tagline;
if (flashMessage) {
  setLandingError(flashMessage);
  localStorage.removeItem("forgensic_flash");
}

updateCtas();

if (auth) {
  getRedirectResult(auth).catch((err) => {
    const code = err?.code || "";
    setLandingError(code ? `Sign-in failed (${code}).` : "Sign-in failed.");
  });
  onAuthStateChanged(auth, (user) => {
    if (!user) return;
    const next = getNextStep();
    if (next === "consent") redirectToConsent();
    if (next === "questionnaire") redirectToQuestionnaire();
    if (next === "app") redirectToApp();
  });
} else {
  updateCtas();
}
