import torch
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from stable_baselines3 import PPO

# --- KONFIGURASI ---
MODEL_PATH = "../models/hybrid_14bitrate_ndn_model_v15.zip"  # Pastikan path ini benar sesuai dengan lokasi model Anda

app = FastAPI(title="NDN-RL Pure Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model = PPO.load(MODEL_PATH, device=device)
    print(f"✅ Model loaded successfully on {device}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

class PredictionRequest(BaseModel):
    observations: list 

@app.post("/predict")
async def predict(metrics: PredictionRequest):
    if model is None:
        return {"bitrate_index": 0, "status": "MODEL_NOT_FOUND"}

    try:
        # 1. Konversi ke Numpy Array (Float32 adalah standar model RL)
        obs_array = np.array(metrics.observations, dtype=np.float32).reshape(1, -1)

        # 2. Prediksi Action
        # action biasanya bertipe numpy.ndarray, misal: array([1])
        action, _states = model.predict(obs_array, deterministic=True)

        # 3. FIX: Gunakan .item() untuk mengambil angka di dalam array secara aman
        bitrate_index = int(action.item())

        print(f"📥 Input: {metrics.observations}")
        print(f"📤 AI Decision: {bitrate_index}")

        return {
            "bitrate_index": bitrate_index,
            "status": "SUCCESS"
        }

    except Exception as e:
        # Log error lebih detail untuk debugging
        print(f"❌ Error saat prediksi: {e}")
        return {"bitrate_index": 0, "status": f"ERROR: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("ndn-rl-inference-server:app", host="0.0.0.0", port=8000, workers=2)