# main.py
import os
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cachetools import TTLCache
from dotenv import load_dotenv

# ──────────────── Cargar variables de entorno ────────────────
load_dotenv()
USE_FIREBASE = os.getenv("USE_FIREBASE", "false").lower() == "true"
FIREBASE_CRED_PATH = os.getenv("FIREBASE_CREDENTIALS", "firebase_credentials.json")

# ──────────────── (Opcional) Inicializar Firebase ────────────────
db = None
if USE_FIREBASE:
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
    except Exception as e:
        # Si hay problema con las credenciales, desactiva Firebase para no romper la API
        print(f"[WARN] No se pudo inicializar Firebase: {e}")
        USE_FIREBASE = False

# ──────────────── Cargar modelo ────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ──────────────── Configurar FastAPI ────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────── Cache y concurrencia ────────────────
cache: TTLCache = TTLCache(maxsize=100, ttl=60)   # ttl en segundos
semaphore = asyncio.Semaphore(5)
executor = ThreadPoolExecutor()

# ──────────────── Esquema de entrada ────────────────
class InputData(BaseModel):
    features: List[int]

# ──────────────── Endpoint ────────────────
@app.post("/predict")
async def predict(data: InputData):
    async with semaphore:
        feats = tuple(data.features)

        if len(feats) != 14:
            return {"error": "Se requieren exactamente 14 características"}

        # ——— Cache ———
        if feats in cache:
            prediction_value = cache[feats]
            cached = True
        else:
            np_features = np.array([data.features])
            prediction = await asyncio.get_event_loop().run_in_executor(
                executor, model.predict, np_features
            )
            prediction_value = int(round(prediction[0]))
            cache[feats] = prediction_value
            cached = False

            # ——— Guardar en Firestore (si está activo) ———
            if USE_FIREBASE and db:
                try:
                    db.collection("predictions").add(
                        {"features": data.features, "prediction": prediction_value}
                    )
                    db_status = "Firestore OK"
                except Exception as e:
                    db_status = f"Firestore Error: {e}"
            else:
                db_status = "Firestore skip"

        return {
            "prediction": prediction_value,
            "cache": cached,
            "db_status": db_status if 'db_status' in locals() else ("Firestore OK" if cached else "Firestore skip"),
        }
