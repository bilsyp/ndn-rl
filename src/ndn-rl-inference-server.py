import numpy as np
import uvicorn
import onnxruntime as ort  # <-- Menggantikan torch & stable-baselines3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

# --- LOCAL PATH CONFIGURATION ---
current_path = Path(__file__).resolve()

# Mencari folder utama (sesuaikan dengan nama folder proyekmu)
ROOT_DIR = next(p for p in current_path.parents if p.name == "membuat-model")

# Diarahkan langsung ke file .onnx hasil ekspor kemarin
# Catatan: Pastikan file .onnx.data berada di folder yang sama dengan file .onnx ini!
MODEL_PATH = ROOT_DIR / "models" / "model-onnx" / "hybrid_14bitrate_ndn_model_v15.onnx"

# --- CONFIGURASI FASTAPI ---
app = FastAPI(title="NDN-RL Pure ONNX Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD ONNX MODEL (SUPER LIGHTWEIGHT) ---
try:
    # ONNX Runtime secara otomatis memilih GPU (CUDA) jika tersedia, jika tidak akan pakai CPU
    # Ini jauh lebih ringan daripada memuat full engine PyTorch
    session = ort.InferenceSession(str(MODEL_PATH))
    
    # Mengambil nama input dan output node dari model ONNX secara dinamis
    input_name = session.get_inputs()[0].name   # Ini akan bernilai "observation"
    output_name = session.get_outputs()[0].name # Ini akan bernilai "action_logits"
    
    print(f"✅ ONNX Model loaded successfully! Providers: {session.get_providers()}")
except Exception as e:
    print(f"❌ Failed to load ONNX model: {e}")
    session = None

class PredictionRequest(BaseModel):
    observations: list 

@app.post("/predict")
async def predict(metrics: PredictionRequest):
    if session is None:
        return {"bitrate_index": 0, "status": "MODEL_NOT_FOUND"}

    try:
        # 1. Konversi input ke Numpy Array Float32 (Wajib Match dengan ONNX Input)
        obs_array = np.array(metrics.observations, dtype=np.float32).reshape(1, -1)

        # 2. Jalankan Inference dengan ONNX Runtime
        # session.run membutuhkan list output_names dan dictionary input data
        raw_outputs = session.run([output_name], {input_name: obs_array})
        
        # 3. Ambil hasil Logits (output berupa list dari numpy array)
        logits = raw_outputs[0] # Bentuknya: [[log1, log2, log3, log4]]

        # 4. Pilih Aksi (Argmax)
        # Karena ONNX mengeluarkan nilai mentah (Logits), kita cari index dengan nilai tertinggi
        bitrate_index = int(np.argmax(logits[0]))

        print(f"📥 Input (NDN) : {metrics.observations}")
        print(f"📤 AI Decision : {bitrate_index}")

        return {
            "bitrate_index": bitrate_index,
            "status": "SUCCESS"
        }

    except Exception as e:
        print(f"❌ Error saat prediksi ONNX: {e}")
        return {"bitrate_index": 0, "status": f"ERROR: {str(e)}"}

if __name__ == "__main__":
    # Tips: uvicorn.run panggil langsung objek 'app' jika ditulis di file yang sama
    uvicorn.run(app, host="0.0.0.0", port=8000)