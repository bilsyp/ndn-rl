import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import deque
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
# =============================================================================
# PERUBAHAN UTAMA DARI VERSI LAMA:
# 1. bitrates = [0.55, 0.95, 1.67, 3.40] → 4 opsi (240p/360p/480p/720p)
# 2. action_space = Discrete(4)
# 3. Scaling trace otomatis ke max_bitrate × 1.5 = 3.40 × 1.5 = 5.10 Mbps
# 4. Normalisasi last_exec dibagi 3.0 (n-1 = 3)
# 5. SAFE_MARGIN diturunkan ke 0.3 (lompatan antar bitrate lebih kecil)
# 6. NDN_CONGESTION_CWND diturunkan ke 3.0 (threshold disesuaikan)
# 7. Total timesteps = 500.000 langkah
# 8. Reward dinormalisasi terhadap max bitrate agar tidak bias
# =============================================================================

# --- KONSTANTA GLOBAL ---
BITRATES       = [0.55, 0.95, 1.67, 3.40]   # Mbps → 240p, 360p, 480p, 720p
MAX_BITRATE    = max(BITRATES)               # 3.40 Mbps
SCALE_TARGET   = MAX_BITRATE * 1.5          # 5.10 Mbps → headroom jaringan
NUM_ACTIONS    = len(BITRATES)              # 4


class MahimahiTraceManager:
    """
    Mengelola file trace Mahimahi (.log).
    Mengonversi timestamp (ms) menjadi Throughput (Mbps).
    Scaling otomatis ke SCALE_TARGET (max_bitrate × 1.5 = 5.10 Mbps)
    agar jaringan cukup menantang untuk 4 level resolusi.
    """
    def __init__(self, folder_path="traces_folder/mahimahi"):
        self.traces = []
        PACKET_SIZE_BITS = 1500 * 8

        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.log')]
            files.sort()

            for file in files:
                path = os.path.join(folder_path, file)
                try:
                    with open(path, 'r') as f:
                        timestamps_ms = [float(line.strip()) for line in f if line.strip()]

                        if timestamps_ms:
                            throughput_mbps = []
                            current_sec = 0
                            packet_count = 0

                            for ts in timestamps_ms:
                                sec = int(ts / 1000)
                                while current_sec < sec:
                                    mbps = (packet_count * PACKET_SIZE_BITS) / 1_000_000
                                    throughput_mbps.append(mbps)
                                    packet_count = 0
                                    current_sec += 1
                                packet_count += 1

                            throughput_mbps.append((packet_count * PACKET_SIZE_BITS) / 1_000_000)

                            # -------------------------------------------------------
                            # PERUBAHAN: Scaling ke SCALE_TARGET (5.10 Mbps)
                            # Sebelumnya hardcoded 12.0 Mbps.
                            # Sekarang otomatis menyesuaikan dengan max bitrate client.
                            # Dengan SCALE_TARGET = 3.40 × 1.5 = 5.10 Mbps:
                            #   - Jaringan "bagus"  → bisa memilih 720p (3.40 Mbps)
                            #   - Jaringan "sedang" → hanya cukup untuk 480p (1.67 Mbps)
                            #   - Jaringan "drop"   → turun ke 360p atau 240p
                            # -------------------------------------------------------
                            max_tp = max(throughput_mbps) if throughput_mbps else 1
                            scale_factor = SCALE_TARGET / max_tp if max_tp > 0 else 1.0

                            scaled_mbps = [max(0.1, tp * scale_factor) for tp in throughput_mbps]

                            # Pre-smoothing ringan (window=3)
                            smoothed = pd.Series(scaled_mbps).rolling(
                                window=3, min_periods=1
                            ).mean().tolist()

                            self.traces.append({"name": file, "data": smoothed})

                except Exception as e:
                    print(f"Gagal memproses file {file}: {e}")

        if not self.traces:
            print("⚠️  traces_folder tidak ditemukan. Gunakan fallback synthetic.")
            # Fallback: throughput sintetis dengan variasi
            synthetic = [3.0 + 1.5 * np.sin(i * 0.1) for i in range(500)]
            self.traces.append({"name": "synth", "data": synthetic})

        self.active_trace = None
        self.ptr = 0

        print(f"✅ {len(self.traces)} trace dimuat. Target scaling: {SCALE_TARGET:.2f} Mbps")

    def select_random_trace(self):
        self.active_trace = random.choice(self.traces)
        self.ptr = random.randint(0, max(0, len(self.active_trace["data"]) - 110))
        return self.active_trace["name"]

    def set_trace_index(self, idx):
        self.active_trace = self.traces[idx % len(self.traces)]
        self.ptr = 0
        return self.active_trace["name"]

    def get_next_bandwidth(self):
        val = self.active_trace["data"][self.ptr]
        self.ptr = (self.ptr + 1) % len(self.active_trace["data"])
        return val


class HybridStreamingEnvNDN(gym.Env):
    """
    Hybrid ABR Environment — versi 4 bitrate (240p / 360p / 480p / 720p).

    Observation Space (7 dimensi, ternormalisasi [0,1]):
      [Buffer, Mean_TP, Last_Exec, Buffer_Safety, Volatility, CWND, RTT]

    Action Space:
      Discrete(4) → indeks 0/1/2/3 → 0.55/0.95/1.67/3.40 Mbps

    Perubahan dari versi 3-bitrate:
      - NUM_ACTIONS = 4, norm_exec dibagi 3.0
      - SCALE_TARGET = 5.10 Mbps (bukan 12 Mbps)
      - SAFE_MARGIN = 0.3 Mbps (lompatan antar level lebih kecil)
      - NDN_CONGESTION_CWND = 3.0 (lebih sensitif)
      - Reward dinormalisasi: reward_bitrate = chosen / MAX_BITRATE
    """

    def __init__(self, trace_manager):
        super(HybridStreamingEnvNDN, self).__init__()
        self.trace_manager = trace_manager

        # -------------------------------------------------------
        # PERUBAHAN UTAMA: 4 level bitrate sesuai resolusi client
        # -------------------------------------------------------
        self.bitrates = BITRATES   # [0.55, 0.95, 1.67, 3.40]

        # Observation: 7 dimensi [0,1]
        self.observation_space = spaces.Box(
            low=np.zeros(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32),
            dtype=np.float32
        )

        # -------------------------------------------------------
        # PERUBAHAN: Discrete(4) — bukan Discrete(3)
        # -------------------------------------------------------
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.state = None
        self.max_steps = 100
        self.current_step = 0
        self.tp_history = deque(maxlen=10)

        # --- Threshold Buffer ---
        self.LOW_BUFFER_THRESHOLD   = 15   # detik → warning kuning
        self.PANIC_BUFFER_THRESHOLD = 5  # detik → panic merah (HARUS < LOW!)

        # -------------------------------------------------------
        # PERUBAHAN: SAFE_MARGIN diturunkan dari 1.5 → 0.3
        # Karena lompatan antar bitrate sekarang lebih kecil:
        #   0.55→0.95 = +0.40 Mbps
        #   0.95→1.67 = +0.72 Mbps
        #   1.67→3.40 = +1.73 Mbps
        # Margin 1.5 Mbps terlalu besar untuk 3 step pertama.
        # -------------------------------------------------------
        self.SAFE_MARGIN = 0.3   # Mbps headroom minimum untuk upgrade

        # -------------------------------------------------------
        # PERUBAHAN: NDN_CONGESTION_CWND diturunkan dari 5.0 → 3.0
        # Karena throughput operasional lebih rendah (maks ~5 Mbps),
        # CWND kritis perlu disesuaikan agar veto tetap efektif.
        # -------------------------------------------------------
        self.NDN_CONGESTION_CWND = 3.0
        self.NDN_CACHE_HIT_RTT   = 20.0

    # ------------------------------------------------------------------
    def _get_normalized_obs(self):
        buffer, mean_tp, last_exec, _, volatility, cwnd, rtt = self.state

        norm_buffer = np.clip(buffer / 30.0, 0.0, 1.0)
        norm_tp     = np.clip(mean_tp / SCALE_TARGET, 0.0, 1.0)  # dibagi 5.10

        # -------------------------------------------------------
        # PERUBAHAN: norm_exec dibagi (NUM_ACTIONS - 1) = 3.0
        # Sebelumnya dibagi 2.0 (untuk Discrete(3))
        # -------------------------------------------------------
        norm_exec = float(last_exec) / float(NUM_ACTIONS - 1)

        safety      = max(0, buffer - self.LOW_BUFFER_THRESHOLD)
        norm_safety = np.clip(safety / 20.0, 0.0, 1.0)
        norm_vol    = np.clip(volatility, 0.0, 1.0)
        norm_cwnd   = np.clip(cwnd / 100.0, 0.0, 1.0)
        norm_rtt    = np.clip(rtt / 500.0, 0.0, 1.0)

        return np.array(
            [norm_buffer, norm_tp, norm_exec, norm_safety, norm_vol, norm_cwnd, norm_rtt],
            dtype=np.float32
        )

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options and "trace_idx" in options:
            trace_name = self.trace_manager.set_trace_index(options["trace_idx"])
        else:
            trace_name = self.trace_manager.select_random_trace()

        initial_tp = self.trace_manager.get_next_bandwidth()

        self.tp_history.clear()
        for _ in range(self.tp_history.maxlen):
            self.tp_history.append(initial_tp)

        # State awal: buffer 15 detik, indeks 1 (360p), CWND 10, RTT 100ms
        self.state = np.array(
            [15.0, initial_tp, 1.0, 0.0, 0.0, 10.0, 100.0],
            dtype=np.float32
        )
        self.current_step = 0
        return self._get_normalized_obs(), {"trace": trace_name}

    # ------------------------------------------------------------------
    def step(self, action):
        if isinstance(action, (np.ndarray, list)):
        # Jika array ([1],), ambil isinya. Jika skalar array(1), juga akan terambil isinya.
         target_idx = int(np.array(action).item())
        else:
         target_idx = int(action)
        # 0. Inisialisasi awal
        buffer, _, current_idx, _, _, prev_cwnd, prev_rtt = self.state
        current_idx = int(current_idx)
        
        # 1. Update throughput & metrik jaringan (Kondisi Lingkungan Saat Ini)
        raw_tp = self.trace_manager.get_next_bandwidth()
        self.tp_history.append(raw_tp)
        mean_tp = np.mean(self.tp_history)
        volatility = np.std(self.tp_history) / (mean_tp + 1e-6)

        # Simulasi metrik NDN (Observasi Lingkungan)
        new_cwnd = np.clip(raw_tp * 8.0 + np.random.normal(0, 2), 2, 100)
        new_rtt  = np.clip(200.0 / (raw_tp + 0.5) + np.random.normal(0, 5), 5, 500)

        # ------------------------------------------------------------------
        # LAYER 2: NDN-AWARE DYNAMIC CONTROLLER (FILTER KEAMANAN)
        # ------------------------------------------------------------------
        # Kita tentukan EXECUTED_IDX di sini, SEBELUM menghitung download_time
        
        executed_idx = current_idx # Default: bertahan di bitrate sekarang
        dynamic_buffer_req = self.LOW_BUFFER_THRESHOLD + (volatility * 5.0)

        # A. PANIC MODE (Prioritas Utama: Cegah Stalling Total)
        if buffer < self.PANIC_BUFFER_THRESHOLD:
            executed_idx = max(0, current_idx - 1)
        
        # B. NDN VETO & UPGRADE LOGIC (Hanya jika tidak dalam Panic Mode)
        else:
            if target_idx > current_idx:
                # Cek apakah jalur macet (Veto)
                is_ndn_congested = new_cwnd < self.NDN_CONGESTION_CWND
                headroom = mean_tp - self.bitrates[current_idx]
                
                # Syarat Upgrade: Tidak macet + Headroom cukup + Buffer aman + RTT rendah
                if not is_ndn_congested and \
                   headroom > self.SAFE_MARGIN and \
                   buffer > dynamic_buffer_req and \
                   new_rtt < 300:
                    executed_idx = current_idx + 1
                else:
                    executed_idx = current_idx # Upgrade ditolak oleh Layer 2
            
            elif target_idx < current_idx:
                # Downgrade: Agen diizinkan turun kapan saja demi efisiensi/keamanan
                executed_idx = target_idx
            
            else:
                # Tetap di bitrate yang sama
                executed_idx = current_idx

        # Pastikan indeks tetap dalam batas valid
        executed_idx = int(np.clip(executed_idx, 0, NUM_ACTIONS - 1))

        # ------------------------------------------------------------------
        # 2. EKSEKUSI FISIK (Berdasarkan executed_idx yang sudah tervalidasi)
        # ------------------------------------------------------------------
        chosen_bitrate = self.bitrates[executed_idx]

        # Simulasi DASH download
        # (Download time dan stalling sekarang SINKRON dengan bitrate yang benar-benar dipakai)
        download_time = (chosen_bitrate * 5.0) / (raw_tp + 0.1)
        stalling      = max(0, download_time - buffer)
        
        # Update Buffer State
        new_buffer = max(0, buffer - download_time) + 5.0
        new_buffer = min(new_buffer, 30.0)

        # ------------------------------------------------------------------
        # 3. REWARD FUNCTION (Berdasarkan hasil nyata eksekusi)
        # ------------------------------------------------------------------
        
        # A. Reward Bitrate (Normalisasi 0.0 - 100.0)
        norm_reward = np.log2(chosen_bitrate / self.bitrates[0])

       # B. Reward buffer (pakai konstanta, bukan magic number)
        if new_buffer < self.PANIC_BUFFER_THRESHOLD:
            reward_buffer = -20.0
        elif new_buffer < self.LOW_BUFFER_THRESHOLD:
            reward_buffer = -8.0
        elif new_buffer > 20.0:
            reward_buffer = 6.0
            if executed_idx == 3:
                reward_buffer += 2.0
        else:
            reward_buffer = 2.0

        # C. Penalti Stalling (Dihitung berdasarkan stalling nyata)
        reward = norm_reward + reward_buffer
        if stalling > 0:
          reward -= (15.0 + stalling * 5.0)
        # D. Penalti Fluktuasi (Smoothness)
        # Dibandingkan dengan current_idx dari state sebelumnya
        if executed_idx != current_idx:
           reward -= 5.0 * abs(executed_idx - current_idx)
        # ------------------------------------------------------------------
        # 4. UPDATE STATE & RETURN
        # ------------------------------------------------------------------
        self.state = np.array(
            [new_buffer, mean_tp, float(executed_idx), 0.0,
             volatility, new_cwnd, new_rtt],
            dtype=np.float32
        )

        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info = {
            "buffer":      new_buffer,
            "cwnd":        new_cwnd,
            "rtt":         new_rtt,
            "bitrate":     chosen_bitrate,
            "resolution":  ["240p", "360p", "480p", "720p"][executed_idx],
            "raw_tp":      mean_tp,
            "volatility":  volatility,
            "stalling":    stalling,
            "is_panic":    buffer < self.PANIC_BUFFER_THRESHOLD
        }

        return self._get_normalized_obs(), reward, done, False, info



def make_env(rank, log_dir, seed=0):
    def _init():
        # 1. Setiap worker membuat manager-nya sendiri
        tm = MahimahiTraceManager(folder_path="traces_folder/mahimahiv2")
        
        # 2. Inisialisasi Environment
        env = HybridStreamingEnvNDN(tm)
        
        # 3. Monitor HARUS ada di sini agar merekam data per worker.
        # Kita beri nama file yang unik berdasarkan 'rank' (0, 1, 2, 3)
        # agar tidak saling menindih (monitor_0.csv, monitor_1.csv, dst)
        monitor_path = os.path.join(log_dir, str(rank))
        env = Monitor(env, monitor_path)
        
        return env
    return _init
# =============================================================================
def run_experiment():
    log_dir = "./rl_logs_6bitrate/"
    os.makedirs(log_dir, exist_ok=True)

    tm = MahimahiTraceManager(folder_path="traces_folder/mahimahiv2/")
    if not tm.traces:
        return

    num_cpu = 4
    total_steps = 1000000
    # Kirim log_dir ke fungsi make_env
    env = SubprocVecEnv([make_env(i, log_dir) for i in range(num_cpu)])

    # --- LANGKAH 3: Model ---
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    print("=" * 60)
    print("  Volatility-Aware Hybrid ABR — 4 Bitrate (240p~720p)")
    print(f"  Bitrates : {BITRATES} Mbps")
    print(f"  TP Scale : {SCALE_TARGET:.2f} Mbps (max_bitrate × 1.5)")
    print(f"  Actions  : Discrete({NUM_ACTIONS})")
    print(f"  Training : {total_steps} langkah")
    print("=" * 60)

    # -------------------------------------------------------
    # PERUBAHAN: total_timesteps = 500_000 (dari 300_000)
    # Action space lebih besar (4 vs 3) butuh lebih banyak
    # pengalaman agar agen menguasai semua transisi.
    # -------------------------------------------------------
    
    # model = PPO(
    #     "MlpPolicy", env,
    #     verbose=1,
    #     learning_rate=3e-4,
    #     ent_coef=0.05,
    #     n_epochs=10,
    #     n_steps=2048,
    #     batch_size=64
    # )
    model = PPO(
    "MlpPolicy", env,
    verbose=1,
    tensorboard_log=log_dir,

    # Learning rate: turunkan + pakai linear decay
    # 3e-4 terlalu agresif untuk environment yang sudah di-shape
    learning_rate=lambda progress: 2.5e-4 * progress,  # decay ke 0 di akhir

    # Entropy: mulai sedang, biarkan decay alami via lr schedule
    # 0.05 terlalu tinggi untuk konvergensi
    ent_coef=0.01,

    # Naikkan n_steps agar satu rollout lebih representatif
    # 2048 × 4 workers = 8192 → naikkan ke 4096 × 4 = 16384
    n_steps=4096,

    # Kurangi epoch: 10 terlalu banyak → overfitting ke rollout
    n_epochs=4,

    # Naikkan batch_size seiring n_steps naik
    # rule of thumb: batch_size ≈ n_steps * n_envs / 16
    batch_size=256,

    # Tambahan yang kamu belum set tapi penting:
    clip_range=0.1,          # default 0.2 terlalu lebar untuk env volatile
    vf_coef=0.5,             # default, tapi eksplisit lebih baik
    max_grad_norm=0.5,       # cegah gradient explosion saat collapse
    gae_lambda=0.95,         # default, cocok untuk episode panjang

    )
    
    model.learn(total_timesteps=total_steps,
        progress_bar=True,
        tb_log_name="PPO_Parallel_1M_30Traces")
    model.save("hybrid_4bitrate_ndn_model_v6")
    print("✅ Pelatihan selesai. Model disimpan: hybrid_4bitrate_ndn_model_v5")

    # ------------------------------------------------------------------
    # EVALUASI per trace file
    # ------------------------------------------------------------------
    resolution_labels = ["240p", "360p", "480p", "720p"]
    colors_res        = ["blue", "green", "orange", "red"]

    # ------------------------------------------------------------------
    # EVALUASI per trace file (Jalur Khusus)
    # ------------------------------------------------------------------
    print("\n📊 Memulai Evaluasi Akhir...")
    
    # 1. Tutup env paralel untuk membebaskan memori
    env.close()

    # 2. Buat environment tunggal untuk evaluasi agar mendukung 'options'
    # Kita menggunakan DummyVecEnv agar format step tetap konsisten dengan model
    eval_env = DummyVecEnv([lambda: HybridStreamingEnvNDN(tm)])
    
    resolution_labels = ["240p", "360p", "480p", "720p"]
    colors_res         = ["blue", "green", "orange", "red"]

    # --- BAGIAN EVALUASI DI DALAM run_experiment() ---
    for i in range(len(tm.traces)):
        # Reset env melalui wrapper VecEnv
        # Karena ini VecEnv, reset() mengembalikan obs sebagai array [1, 7]
        obs = eval_env.reset() 
        
        # Akses environment internal untuk menset trace (karena VecEnv tidak mendukung 'options' secara native)
        # Kita panggil reset manual pada internal env untuk mengganti trace
        eval_env.envs[0].reset(options={"trace_idx": i})
        
        trace_name = tm.traces[i]["name"]
        history = []

        for _ in range(100):
            # 1. Prediksi Action - Mengembalikan array, misal: array([1])
            action, _ = model.predict(obs, deterministic=True)
            
            # 2. MASUKKAN LANGSUNG KE STEP (Jangan dibungkus [action])
            # SB3 sudah menangani pemetaan array([1]) ke environment ke-0
            obs, rewards, dones, infos = eval_env.step(action)
            
            # 3. Ambil data info dari worker pertama
            info_step = infos[0]
            
            history.append({
                "TP":         info_step["raw_tp"],
                "Volatility": info_step["volatility"],
                "Buffer":      info_step["buffer"],
                "Bitrate":     info_step["bitrate"],
                "Resolution": info_step["resolution"],
                "Stalling":    info_step["stalling"],
            })
            
            if dones[0]:
                break

        # --- Bagian Plotting (Tetap Menggunakan DataFrame) ---
        df = pd.DataFrame(history)
        fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

        # Panel 1: Throughput vs Bitrate
        ax1 = axes[0]
        ax1.plot(df.index, df["TP"], label="Mean TP", color="steelblue", alpha=0.4)
        ax1.step(df.index, df["Bitrate"], label="Executed Bitrate",
                 color="crimson", linewidth=2, where="post")

        # Gunakan BITRATES dari scope global atau class
        for br, label, col in zip(eval_env.envs[0].bitrates, resolution_labels, colors_res):
            ax1.axhline(y=br, linestyle="--", linewidth=0.8, alpha=0.5,
                        color=col, label=f"{label} ({br} Mbps)")

        ax1.set_ylabel("Mbps")
        ax1.set_title(f"Evaluasi Model: {trace_name}")
        ax1.legend(fontsize=7, ncol=3)

        # Panel 2: Buffer & Stalling
        ax2 = axes[1]
        ax2.fill_between(df.index, df["Buffer"], color="mediumseagreen",
                         alpha=0.3, label="Buffer (detik)")
        ax2.axhline(y=15.0, color="orange", linestyle="--", label="Low Buffer")
        ax2.axhline(y=5.0, color="red", linestyle="--", label="Panic") # Sesuai PANIC_THRESHOLD Anda

        stall_mask = df["Stalling"] > 0
        if stall_mask.any():
            ax2.scatter(df.index[stall_mask], df["Buffer"][stall_mask],
                        color="red", zorder=5, s=30, label="Stalling Event")
        ax2.set_ylabel("Buffer (detik)")
        ax2.legend(fontsize=7)

        # Panel 3: Volatility
        ax3 = axes[2]
        ax3.plot(df.index, df["Volatility"], color="purple", linestyle=":", label="CV")
        ax3.set_ylabel("Volatility")
        ax3.set_xlabel("Segmen")
        ax3.legend(fontsize=7)

        plt.tight_layout()
        out_path = os.path.join(log_dir, f"eval_v2_{trace_name}.png")
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"  ✅ Plot disimpan: {out_path}")

if __name__ == "__main__":
    run_experiment()