import json
import os
from pathlib import Path
from typing import Optional

import firebase_admin
from firebase_admin import auth as fb_auth
from firebase_admin import credentials, firestore, storage

from .config import FIREBASE_CREDENTIALS, FIREBASE_CREDENTIALS_PATH, FIREBASE_STORAGE_BUCKET

_firebase_app = None


def init_firebase() -> Optional[firebase_admin.App]:
    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app

    cred = None
    if FIREBASE_CREDENTIALS:
        cred_dict = json.loads(FIREBASE_CREDENTIALS)
        cred = credentials.Certificate(cred_dict)
    elif FIREBASE_CREDENTIALS_PATH:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        cred = credentials.ApplicationDefault()
    else:
        local_path = Path(__file__).resolve().parents[2] / "forgensic-e3fb6-firebase-adminsdk-fbsvc-d4e2b99c36.json"
        if local_path.exists():
            cred = credentials.Certificate(str(local_path))

    if cred is None:
        return None

    options = {}
    if FIREBASE_STORAGE_BUCKET:
        options["storageBucket"] = FIREBASE_STORAGE_BUCKET

    _firebase_app = firebase_admin.initialize_app(cred, options or None)
    return _firebase_app


def get_firestore():
    app = init_firebase()
    if app is None:
        return None
    return firestore.client(app=app)


def get_storage_bucket():
    app = init_firebase()
    if app is None:
        return None
    try:
        return storage.bucket(app=app)
    except Exception:
        return None


def verify_id_token(token: str):
    app = init_firebase()
    if app is None:
        return None
    try:
        return fb_auth.verify_id_token(token, app=app)
    except Exception:
        return None
