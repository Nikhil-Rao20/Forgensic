import { FIREBASE_CONFIG, APP_BRAND } from "./config.js";
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

async function signInWithGoogle() {
  if (!auth || authBusy) {
    redirectToApp();
    return;
  }
  authBusy = true;
  setLandingError("");
  setAuthButtonsEnabled(false);
  const provider = new GoogleAuthProvider();
  try {
    await signInWithPopup(auth, provider);
    redirectToApp();
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
landingStartBtn?.addEventListener("click", () => signInWithGoogle());

if (brandName) brandName.textContent = APP_BRAND.name;
if (brandSub) brandSub.textContent = APP_BRAND.tagline;
if (flashMessage) {
  setLandingError(flashMessage);
  localStorage.removeItem("forgensic_flash");
}

if (auth) {
  getRedirectResult(auth).catch((err) => {
    const code = err?.code || "";
    setLandingError(code ? `Sign-in failed (${code}).` : "Sign-in failed.");
  });
  onAuthStateChanged(auth, (user) => {
    if (user) redirectToApp();
  });
} else {
  if (landingLoginBtn) landingLoginBtn.textContent = "Enter demo";
  if (landingStartBtn) landingStartBtn.textContent = "Open console";
}
