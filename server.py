from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from stable_baselines3 import PPO
import numpy as np
import uvicorn
import os
import asyncio
from typing import Dict, Tuple

# Inisialisasi FastAPI
app = FastAPI(title="NDN-RL Optimized Tandon Inference")

# Konfigurasi Model
MODEL_PATH = "ndn_video_brain_tandon_multi_trace.zip"
model = None

# Memori Per-Client untuk menghitung Tren (Stateless to Stateful)
# Format: { client_id: (last_buffer, last_throughput) }
client_memories: Dict[str, Tuple[float, float]] = {}

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        # Load model secara sinkron di startup
        model = PPO.load(MODEL_PATH)
        print(f"✅ Model Final '{MODEL_PATH}' berhasil dimuat ke memori.")
    else:
        print(f"❌ ERROR: File '{MODEL_PATH}' tidak ditemukan! Pastikan sudah menjalankan training.")

class Metrics(BaseModel):
    buffer: float
    throughput: float
    last_quality: int
    rtt: float
    dropped_frames: int

class RequestData(BaseModel):
    client_id: str
    next_seg: int
    metrics: Metrics

@app.post("/predict")
async def predict_bitrate(req: RequestData):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum siap atau tidak ditemukan.")

    try:
        cid = req.client_id
        m = req.metrics

        # 1. Ambil data lama dari memori untuk hitung Tren
        # Jika client baru, asumsikan tren 0 (statis)
        last_buffer, last_tp = client_memories.get(cid, (m.buffer, m.throughput))

        # 2. Hitung Tren (Sesuai Logika Tandon Air)
        buf_trend = np.clip(m.buffer - last_buffer, -5.0, 5.0)
        tp_trend = np.clip(m.throughput - last_tp, -5.0, 5.0)

        # 3. Update Memori Client untuk request berikutnya
        client_memories[cid] = (m.buffer, m.throughput)

        # 4. Susun State (7 Variabel sesuai Training Fase 2)
        # [Buffer, Throughput, LastQual, RTT, Dropped, BufTrend, TPTrend]
        state = np.array([
            m.buffer, 
            m.throughput, 
            float(m.last_quality), 
            m.rtt, 
            float(m.dropped_frames),
            buf_trend,
            tp_trend
        ], dtype=np.float32)

        # 5. Eksekusi Prediksi
        # Gunakan model.predict. Kita bisa membungkusnya di thread jika model sangat berat,
        # tapi untuk PPO MLP biasanya sangat cepat (< 1ms).
        action, _ = model.predict(state, deterministic=True)
        
        quality_names = ["LOW", "MID", "HIGH"]
        print(f"📊 [Client: {cid}] Seg: {req.next_seg} | State: [B:{m.buffer}s, T:{m.throughput}M] | Action: {quality_names[int(action)]}")

        return {
            "action": int(action),
            "status": "success",
            "trends": {"buffer": float(buf_trend), "throughput": float(tp_trend)}
        }

    except Exception as e:
        print(f"❌ Prediction Error: {str(e)}")
        # Fallback ke kualitas MEDIUM jika terjadi error teknis
        return {"action": 1, "status": "error_fallback"}

# Endpoint untuk membersihkan memori client yang sudah tidak aktif (opsional)
@app.delete("/memory/{client_id}")
async def clear_memory(client_id: str):
    if client_id in client_memories:
        del client_memories[client_id]
        return {"status": "cleared"}
    return {"status": "not_found"}

if __name__ == "__main__":
    # Optimasi Uvicorn:
    # - workers=1: Untuk riset RL, 1 worker lebih stabil menjaga state model di memori.
    # - timeout_keep_alive: Menjaga socket tetap terbuka untuk request berulang dari Node.js.
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=30, access_log=False)