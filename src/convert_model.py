"""
export_to_onnx.py
=================
Utility untuk mengekspor model PPO (stable-baselines3) ke format ONNX.

Kenapa tidak bisa langsung torch.onnx.export(model.policy, ...)?
  - model.policy adalah ActorCriticPolicy — forward() mengembalikan TUPLE
    (actions, values, log_probs), bukan single tensor.
  - torch.onnx.export butuh output yang flat dan deterministik.
  - Untuk deploy/inference, kita hanya butuh ACTOR (action prediction),
    bukan value head (critic).

Solusi: bungkus bagian actor dalam PolicyActorWrapper sebelum export.

Penggunaan:
  python export_to_onnx.py

Output:
  - models/hybrid_4bitrate_ndn_model_v17.onnx   (model NDN, obs_dim=6)
  - models/http_baseline_model_v1.onnx           (model HTTP, obs_dim=5)

Cara inference tanpa PyTorch (gunakan onnxruntime):
  pip install onnxruntime
  import onnxruntime as ort, numpy as np
  sess = ort.InferenceSession("model.onnx")
  obs = np.array([[...]], dtype=np.float32)  # shape (1, obs_dim)
  logits = sess.run(None, {"observation": obs})[0]
  action = np.argmax(logits)
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from pathlib import Path

# Cari lokasi file saat ini
current_path = Path(__file__).resolve()

# Cari folder bernama 'app-python' di jalur ke belakang (ke atas)
# Kode ini akan otomatis mundur sampai ketemu folder dengan nama tersebut
ROOT_DIR = next(p for p in current_path.parents if p.name == "membuat-model")

# Sekarang kamu tinggal gabungkan ke folder tujuan
MODEL_PATH = ROOT_DIR / "models" / "hybrid_14bitrate_ndn_model_v15"


OUTPUT_FOLDER = ROOT_DIR / "models" / "model-onnx"

# SANGAT PENTING: Bikin foldernya otomatis kalau belum ada biar gak error
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# 3. Tentukan nama file ONNX-nya
OUTPUT_PATH = OUTPUT_FOLDER / "hybrid_14bitrate_ndn_model_v15.onnx"
# =============================================================================
# WRAPPER: Ambil HANYA bagian Actor dari ActorCriticPolicy
# =============================================================================

class PolicyActorWrapper(nn.Module):
    """
    Membungkus ActorCriticPolicy SB3 dan mengekspos hanya logit aksi (actor).

    Forward pass:
      Input  : obs tensor shape (batch, obs_dim)  — float32, ternormalisasi [0,1]
      Output : action_logits shape (batch, n_actions) — float32 (sebelum softmax)

    Untuk mendapatkan aksi deterministik: action = argmax(logits)
    Untuk mendapatkan probabilitas      : probs  = softmax(logits)
    """

    def __init__(self, policy):
        super().__init__()
        self.mlp_extractor  = policy.mlp_extractor   # shared MLP
        self.action_net     = policy.action_net       # linear head → logits

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 1. Ekstrak fitur lewat MLP bersama (pi + vf)
        latent_pi, _ = self.mlp_extractor(obs)
        # 2. Lewatkan ke action head → logits
        action_logits = self.action_net(latent_pi)
        return action_logits


# =============================================================================
# FUNGSI EXPORT UTAMA
# =============================================================================

def export_ppo_to_onnx(
    model_path: str,
    output_path: str,
    obs_dim: int,
    opset_version: int = 17
):
    """
    Memuat model PPO dari file .zip dan mengekspornya ke ONNX.

    Args:
        model_path   : path ke file .zip model SB3 (tanpa ekstensi .zip)
        output_path  : path output file .onnx
        obs_dim      : dimensi observation space
                         6 untuk model NDN (dengan CWND)
                         5 untuk model HTTP baseline (tanpa CWND)
        opset_version: ONNX opset (default 17, min 12)
    """
    print(f"\n{'='*55}")
    print(f"  Ekspor ONNX: {model_path}")
    print(f"  obs_dim    : {obs_dim}")
    print(f"  output     : {output_path}")
    print(f"{'='*55}")

    # 1. Muat model SB3
    model = PPO.load(model_path)
    print("  ✅ Model SB3 dimuat.")

    # 2. Verifikasi obs_dim sesuai
    loaded_obs_dim = model.observation_space.shape[0]
    if loaded_obs_dim != obs_dim:
        raise ValueError(
            f"obs_dim tidak cocok! "
            f"Model punya {loaded_obs_dim} dimensi, kamu kasih {obs_dim}. "
            f"Sesuaikan parameter obs_dim."
        )

    # 3. Buat wrapper & set ke eval mode
    policy  = model.policy
    wrapper = PolicyActorWrapper(policy)
    wrapper.eval()
    print("  ✅ PolicyActorWrapper dibuat, mode eval.")

    # 4. Buat dummy input shape (1, obs_dim) — sesuai batch inference
    dummy_input = torch.zeros(1, obs_dim, dtype=torch.float32)

    # 5. Dry-run untuk verifikasi sebelum export
    with torch.no_grad():
        test_out = wrapper(dummy_input)
    print(f"  ✅ Dry-run OK. Output shape: {test_out.shape} "
          f"(expected [1, {model.action_space.n}])")

  # Membuat dimensi batch menjadi dinamis
    batch_dim = torch.export.Dim("batch_size", min=1, max=1024)
    
    # Karena dummy_input cuma ada 1, kita petakan langsung posisinya lewat list/tuple
    # Struktur di dalam list ini harus mencerminkan struktur dummy_input kamu
    dynamic_shapes = [{0: batch_dim}] 

    torch.onnx.export(
        wrapper,
        (dummy_input,),  
        output_path,
        opset_version=18,  
        input_names=["observation"],
        output_names=["action_logits"],
        dynamic_shapes=dynamic_shapes, # Menggunakan format list posisi
    )
    print(f"  ✅ ONNX tersimpan: {output_path}\n")

    return output_path


# =============================================================================
# VERIFIKASI ONNX (opsional, butuh onnxruntime)
# =============================================================================

def verify_onnx_output(onnx_path: str, obs_dim: int, n_actions: int = 4):
    """
    Verifikasi bahwa file ONNX bisa dijalankan dan outputnya masuk akal.
    Butuh: pip install onnxruntime
    """
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path)
        dummy_obs = np.random.rand(1, obs_dim).astype(np.float32)
        logits = sess.run(None, {"observation": dummy_obs})[0]
        action = int(np.argmax(logits))

        print(f"  🔍 Verifikasi ONNX: {onnx_path}")
        print(f"     Input shape : {dummy_obs.shape}")
        print(f"     Logits      : {logits}")
        print(f"     Action      : {action} (argmax)")
        print(f"  ✅ Verifikasi berhasil!\n")

    except ImportError:
        print("  ⚠️  onnxruntime tidak terinstal. "
              "Jalankan: pip install onnxruntime")
    except Exception as e:
        print(f"  ❌ Verifikasi gagal: {e}")


# =============================================================================
# CONTOH INFERENCE TANPA PYTORCH (untuk deploy di Bridge/server ringan)
# =============================================================================

def inference_example():
    """
    Contoh cara menggunakan model ONNX saat deploy.
    Tidak perlu import torch, stable_baselines3, atau gymnasium.
    Hanya butuh: numpy + onnxruntime
    """
    try:
        import onnxruntime as ort

        BITRATES = [0.55, 0.95, 1.67, 3.40]

        print("=" * 55)
        print("  Contoh Inference Pure ONNX (tanpa PyTorch)")
        print("=" * 55)

        # Muat sesi ONNX (lakukan sekali saat startup)
        sess = ort.InferenceSession(OUTPUT_PATH)

        # Simulasi observation dari sensor (ternormalisasi [0,1])
        # Urutan: [Buffer, Mean_TP, Last_Exec, Buffer_Safety, Volatility, CWND]
        # Untuk HTTP baseline (5 dim), hapus elemen terakhir (CWND)
        obs_ndn = np.array([[
            0.70,   # norm_buffer      (buffer ~21s dari max 30s)
            0.55,   # norm_tp          (throughput ~2.8 Mbps dari 5.1 Mbps)
            0.33,   # norm_exec        (aksi terakhir = indeks 1 dari 3)
            0.40,   # norm_safety      (buffer_safety moderat)
            0.15,   # norm_volatility  (jaringan cukup stabil)
            0.60,   # norm_cwnd        (CWND ~30.6 dari headroom NDN)
        ]], dtype=np.float32)

        logits = sess.run(None, {"observation": obs_ndn})[0]
        action = int(np.argmax(logits))

        print(f"  Observation (NDN) : {obs_ndn[0]}")
        print(f"  Logits            : {logits[0]}")
        print(f"  Action dipilih    : {action} → {BITRATES[action]} Mbps\n")

    except Exception as e:
        print(f"  Contoh inference gagal: {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # --- Export Model NDN (6 dimensi, dengan CWND) ---
    export_ppo_to_onnx(
        model_path=MODEL_PATH,   # tanpa .zip
        output_path=OUTPUT_PATH,
        obs_dim=6,
    )

    # # --- Export Model HTTP Baseline (5 dimensi, tanpa CWND) ---
    # export_ppo_to_onnx(
    #     model_path="http_baseline_4bitrate_model_v1",
    #     output_path="http_baseline_4bitrate_model_v1.onnx",
    #     obs_dim=5,
    # )
 
    # # --- Verifikasi kedua model ---
    # verify_onnx_output("hybrid_4bitrate_ndn_model_v16.onnx",     obs_dim=6)
    # verify_onnx_output("http_baseline_4bitrate_model_v1.onnx",   obs_dim=5)

    # --- Contoh inference tanpa PyTorch ---
    inference_example()