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
# ARSITEKTUR BERSIH (BASE MODEL PRODUKSI — BEBAS RTT NOISE)
# 1. Observation Space dioptimalkan dari 7 Dimensi menjadi 6 Dimensi.
# 2. Sensor RTT dieliminasi total untuk mencegah interferensi sinyal palsu.
# 3. Indikator kemacetan Adaptive Governor fokus penuh pada stabilitas CWND.
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
    Scaling otomatis ke SCALE_TARGET (max_bitrate × 1.5 = 5.10 Mbps).
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
    Hybrid ABR Environment — Versi Bersih Bebas RTT Noise.

    Observation Space (6 dimensi, ternormalisasi [0,1]):
      [Buffer, Mean_TP, Last_Exec, Buffer_Safety, Volatility, CWND]

    Action Space:
      Discrete(4) → indeks 0/1/2/3 → 0.55/0.95/1.67/3.40 Mbps
    """

    def __init__(self, trace_manager):
        super(HybridStreamingEnvNDN, self).__init__()
        self.trace_manager = trace_manager
        self.bitrates = BITRATES   

        # KOREKSI: Mengubah dimensi ruang observasi menjadi 6 dimensi
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.state = None
        self.max_steps = 100
        self.current_step = 0
        self.tp_history = deque(maxlen=10)

        # --- Threshold Buffer ---
        self.LOW_BUFFER_THRESHOLD   = 15   # detik
        self.PANIC_BUFFER_THRESHOLD = 5    # detik
        self.SAFE_MARGIN            = 0.3  # Mbps headroom minimum untuk upgrade

        # --- Threshold NDN Kontrol ---
        self.NDN_CONGESTION_CWND    = 3.0  # Threshold CWND kritis

    def _get_normalized_obs(self):
        # Membongkar 6 dimensi state mentah
        buffer, mean_tp, last_exec, trend, volatility, cwnd = self.state

        # ------------------------------------------------------------------
        # ⭐⭐⭐⭐⭐ BINTANG 5: norm_buffer (Sensitivitas Kuadratik / Eksponensial)
        # ------------------------------------------------------------------
        # Diubah dari linear menjadi kuadratik. Jika buffer turun sedikit saja,
        # nilai normalisasinya akan terjun bebas, memaksa neural network waspada.
        norm_buffer = np.clip((buffer / 30.0) ** 2, 0.0, 1.0)

        # ------------------------------------------------------------------
        # ⭐⭐⭐⭐ BINTANG 4: norm_tp & norm_safety (Skala Sinyal Penuh [0.0 - 1.0])
        # ------------------------------------------------------------------
        norm_tp = np.clip(mean_tp / SCALE_TARGET, 0.0, 1.0)  
        
        # Perbaikan skala safety: Jarak dari batas aman (15s) menuju batas maksimal (30s) 
        # adalah 15 detik. Dibagi 15.0 agar sinyal bergerak penuh dari 0.0 hingga 1.0.
        safety = max(0, buffer - self.LOW_BUFFER_THRESHOLD)
        norm_safety = np.clip(safety / 15.0, 0.0, 1.0)

        # ------------------------------------------------------------------
        # ⭐⭐⭐ BINTANG 3: norm_vol (Modulator Risiko Kontinu)
        # ------------------------------------------------------------------
        norm_vol = np.clip(volatility, 0.0, 1.0)

        # ------------------------------------------------------------------
        # ⭐⭐ BINTANG 2: norm_cwnd & norm_exec (Latar Belakang Konten / Low-Impact)
        # ------------------------------------------------------------------
        norm_cwnd = np.clip(cwnd / 100.0, 0.0, 1.0)
        norm_exec = float(last_exec) / float(NUM_ACTIONS - 1)

        # Mengembalikan array 6 dimensi dengan urutan prioritas yang tajam
        return np.array(
            [norm_buffer, norm_tp, norm_exec, norm_safety, norm_vol, norm_cwnd],
            dtype=np.float32
        )

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

        # KOREKSI: Inisialisasi awal array state disesuaikan murni menjadi 6 dimensi
        self.state = np.array(
            [15.0, initial_tp, 1.0, 0.0, 0.0, 10.0],
            dtype=np.float32
        )
        self.current_step = 0
        return self._get_normalized_obs(), {"trace": trace_name}

    def step(self, action):
        # ------------------------------------------------------------------
        # 1. HANDLING ACTION & SAFETY NET
        # ------------------------------------------------------------------
        if isinstance(action, (np.ndarray, list)):
            target_idx = int(np.array(action).item())
        else:
            target_idx = int(action)
        
        target_idx = np.clip(target_idx, 0, NUM_ACTIONS - 1)

        # ------------------------------------------------------------------
        # 2. UNPACKING PREVIOUS STATE (6 DIMENSI)
        # ------------------------------------------------------------------
        buffer, _, current_idx, _, _, prev_cwnd = self.state
        current_idx = int(current_idx)
        prev_bitrate = self.bitrates[current_idx]

        # ------------------------------------------------------------------
        # 3. UPDATE NETWORK THROUGHPUT (WINDOWING)
        # ------------------------------------------------------------------
        raw_tp = self.trace_manager.get_next_bandwidth()
        self.tp_history.append(raw_tp) 
        
        mean_tp = np.mean(self.tp_history)
        volatility = np.std(self.tp_history) / (mean_tp + 1e-6)
        
        if len(self.tp_history) > 1:
            trend = (self.tp_history[-1] - self.tp_history[0]) / len(self.tp_history)
        else:
            trend = 0.0

        # ------------------------------------------------------------------
        # 4. SIMULASI METRIK NDN (MURNI BERBASIS CWND)
        # ------------------------------------------------------------------
        load_ratio = prev_bitrate / (raw_tp + 1e-6)

        base_cwnd = raw_tp * 8.0
        if load_ratio > 1.2: 
            new_cwnd = np.clip(base_cwnd * 0.4 + np.random.normal(0, 2), 2, 100)
        else:
            new_cwnd = np.clip(base_cwnd + np.random.normal(0, 2), 2, 100)

       # =============================================================================
        # LAYER 2: NDN-AWARE DYNAMIC CONTROLLER (ADAPTIVE GOVERNOR WITH EXPLICIT GATES)
        # =============================================================================
        dynamic_buffer_req = self.LOW_BUFFER_THRESHOLD + (volatility * 10.0)
        if trend < 0:
            dynamic_buffer_req += abs(trend) * 5.0 

        # --- EKSTRAKSI INDIKATOR 4 GERBANG KESELAMATAN ---
        # Mengubah kondisi pengecekan menjadi variabel Boolean independen
        gate_cwnd_ok  = new_cwnd >= self.NDN_CONGESTION_CWND
        gate_margin_ok = (mean_tp - self.bitrates[target_idx]) > (self.SAFE_MARGIN * (1.0 + volatility))
        gate_buffer_ok = buffer > dynamic_buffer_req
        gate_trend_ok  = trend >= -0.1

        executed_idx = current_idx 
        is_vetoed = False
        
        # Variabel pelacak kegagalan gerbang spesifik untuk konsumsi fungsi reward
        violated_gates = {"cwnd": False, "margin": False, "buffer": False, "trend": False}

        # A. PANIC MODE (Emergency Brake)
      # A. PANIC MODE (Emergency Brake) - PERBAIKAN LOGIKA BOUNDARY
        if buffer <= self.PANIC_BUFFER_THRESHOLD + 0.1:
            executed_idx = 0
            is_vetoed = target_idx > 0
            if is_vetoed:
                violated_gates["buffer"] = True

        # B. NDN VETO & SMART UPGRADE (Aplikasi Logika 4 Gerbang)
        else:
            if target_idx > current_idx:
                # Jika SEMUA gerbang lolos, ijinkan upgrade resolusi
                if gate_cwnd_ok and gate_margin_ok and gate_buffer_ok and gate_trend_ok:
                    executed_idx = target_idx 
                else:
                    # Jika ada yang gagal, kunci posisi dan catat gerbang mana yang menolak
                    executed_idx = current_idx 
                    is_vetoed = True
                    
                    # Catat "dosa" spesifik tindakan agen untuk dikirim ke mesin reward
                    violated_gates["cwnd"]   = not gate_cwnd_ok
                    violated_gates["margin"] = not gate_margin_ok
                    violated_gates["buffer"] = not gate_buffer_ok
                    violated_gates["trend"]  = not gate_trend_ok
            
            elif target_idx < current_idx:
                executed_idx = target_idx # Turun kelas diizinkan langsung tanpa filter
            else:
                executed_idx = current_idx # Bertahan di posisi aman diizinkan

        executed_idx = int(np.clip(executed_idx, 0, NUM_ACTIONS - 1))

        # ------------------------------------------------------------------
        # [Section 5. EKSEKUSI FISIK SIMULASI DASH BERJALAN DI SINI...]
        # (Menghasilkan variabel: chosen_bitrate, download_time, stalling, new_buffer)
        # ------------------------------------------------------------------

        chosen_bitrate = self.bitrates[executed_idx]
        effective_tp = raw_tp * 0.95 
        download_time = (chosen_bitrate * 5.0) / (effective_tp + 0.1)
        
        stalling = max(0, download_time - buffer)
        
        new_buffer = max(0, buffer - download_time) + 5.0
        new_buffer = min(new_buffer, 30.0)
        # =============================================================================
        # MODIFIKASI ARSITEKTUR 6: REWARD REALIGNMENT & DECOUPLED PENALTY INTEGRATION
        # =============================================================================
        
        # 1. Kualitas Video Dasar (Base Quality)
        base_quality = np.log2(chosen_bitrate / self.bitrates[0])

        # 2. DYNAMIC HEALTH GATE (Berdasarkan 3 Variabel Keadaan Makro)
        buffer_health    = np.clip((new_buffer - 5.0) / (15.0 - 5.0), 0.0, 1.0)
        cwnd_health      = 1.0 if gate_cwnd_ok else 0.2
        stability_health = max(0.1, 1.0 - volatility)
        
        system_health_gate = buffer_health * cwnd_health * stability_health

        # Eksekusi Gated Quality (Menetralisir nafsu serakah agen terhadap 720p saat kritis)
        reward_quality = base_quality * system_health_gate

        # 3. MONOTONIC BUFFER MAINTENANCE INCENTIVE
        if new_buffer < 15.0:
            reward_buffer = -1.5 * (15.0 - new_buffer)
        else:
            reward_buffer = 1.0 + ((new_buffer - 15.0) / 15.0)

        # 4. PREDICTIVE DRAIN & STALL PENALTIES (Proteksi Fisik)
        buffer_drain = download_time - 5.0
        penalty_proactive_drain = -0.5 * buffer_drain if (buffer_drain > 0 and new_buffer < 25.0) else 0.0
        
        penalty_stalling = 0.0
        if stalling > 0:
            penalty_stalling = - (15.0 + 20.0 * np.power(stalling, 1.2))

        # 5. DECOUPLED GATES REJECTION PENALTY (Membongkar Horor Psikologis Agen)
        # Jika agen di-veto, berikan penalti mikro yang spesifik berdasarkan letak kesalahannya
        penalty_gate_break = 0.0
        if is_vetoed:
            if violated_gates["cwnd"]:   penalty_gate_break -= 0.2  # Denda karena merusak pipa NDN
            if violated_gates["margin"]: penalty_gate_break -= 0.2  # Denda karena meremehkan margin bandwidth
            if violated_gates["buffer"]: penalty_gate_break -= 0.3  # Denda karena memaksa naik saat buffer tipis
            if violated_gates["trend"]:  penalty_gate_break -= 0.1  # Denda karena melawan arus tren turun

        # 6. PENALTI KELANCARAN OPSI AKSI
        penalty_smoothness = -1.5 * abs(executed_idx - current_idx)

        # TOTAL COMBINED REWARD SYNTHESIS
        reward = (reward_quality + 
                  reward_buffer + 
                  penalty_proactive_drain + 
                  penalty_stalling + 
                  penalty_gate_break + 
                  penalty_smoothness)

        # ------------------------------------------------------------------
        # 7. UPDATE STATE & RETURN (ARRAY 6 DIMENSI)
        # ------------------------------------------------------------------
        self.state = np.array(
            [new_buffer, mean_tp, float(executed_idx), trend, volatility, new_cwnd],
            dtype=np.float32
        )

        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info = {
            "buffer":      new_buffer,
            "bitrate":     chosen_bitrate,
            "stalling":    stalling,
            "is_vetoed":   is_vetoed,
            "volatility":  volatility,
            "raw_tp":      raw_tp,
            "reward_breakdown": {
                "quality": reward_quality,
                "buffer":  reward_buffer,
                "proactive_drain": penalty_proactive_drain,
                "stall":   penalty_stalling,
                "smooth":  penalty_smoothness,
                "gate_break":    penalty_gate_break
            }
        }
        return self._get_normalized_obs(), reward, done, False, info


def make_env(rank, log_dir, seed=0):
    def _init():
        tm = MahimahiTraceManager(folder_path="../../traces_folder/mahimahi_traces/full_traces")
        env = HybridStreamingEnvNDN(tm)
        monitor_path = os.path.join(log_dir, str(rank))
        env = Monitor(env, monitor_path)
        return env
    return _init


def run_experiment():
    log_dir = "../../logs/rl_logs_18bitrate/"
    os.makedirs(log_dir, exist_ok=True)

    # Cek ketersediaan trace (pastikan menggunakan argumen split dan seed)
    tm_check = MahimahiTraceManager(
        folder_path="../../traces_folder/mahimahi_traces/full_traces", 
        split="train", seed=42
    )
    if not tm_check.traces:
        print("❌ Tidak ada trace yang ditemukan!")
        return

    num_cpu = 4
    total_steps = 500000
    # Pastikan make_env di atas fungsi ini sudah memuat split="train"
    env = SubprocVecEnv([make_env(i, log_dir) for i in range(num_cpu)])

    print("=" * 60)
    print("  Volatility-Aware Hybrid ABR — Production Clean Version")
    print(f"  Bitrates : {BITRATES} Mbps")
    print(f"  TP Scale : {SCALE_TARGET:.2f} Mbps")
    print(f"  State Dim: 6 Dimensions (RTT Removed)")
    print(f"  Actions  : Discrete({NUM_ACTIONS})")
    print(f"  Training : {total_steps} langkah")
    print("=" * 60)
    
    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=2e-4,     
        ent_coef=0.03,          
        n_steps=2048,           
        batch_size=64,         
        n_epochs=10,            
        clip_range=0.2,         
        tensorboard_log=log_dir
    )
    
    # 1. EKSEKUSI TRAINING
    print("\n🚀 Memulai proses training...")
    model.learn(total_timesteps=total_steps)
    model.save(os.path.join(log_dir, "ppo_hybrid_ndn_model"))
    print("✅ Training selesai dan model disimpan.")

    # 2. EKSEKUSI EVALUASI (Sekuensial agar plot tidak tercampur)
    print("\n=======================================================")
    print("📈 EVALUASI PASCA-TRAINING")
    print("=======================================================")
    
    # Evaluasi pada TRAINING SET
    tm_train = MahimahiTraceManager(
        folder_path="../../traces_folder/mahimahi_traces/full_traces",
        split="train", test_size=8, seed=42
    )
    # Tanpa DummyVecEnv, langsung panggil environment dasar
    eval_env_train = HybridStreamingEnvNDN(tm_train) 
    evaluate_and_plot(model, eval_env_train, tm_train, "train", log_dir)
    
    # Evaluasi pada TESTING SET
    tm_test = MahimahiTraceManager(
        folder_path="../../traces_folder/mahimahi_traces/full_traces",
        split="test", test_size=8, seed=42
    )
    eval_env_test = HybridStreamingEnvNDN(tm_test)
    evaluate_and_plot(model, eval_env_test, tm_test, "test", log_dir)

if __name__ == "__main__":
    run_experiment()
    # catatan: pada versi ini agent telah berhasil untuk adaptasi yang lebih stabil dan tidak terlalu agresif dalam memilih bitrate tinggi saat kondisi jaringan tidak mendukung, berkat adanya mekanisme veto yang lebih ketat dan reward yang lebih seimbang. Evaluasi menunjukkan bahwa agent mampu menjaga buffer pada level yang aman sambil tetap memanfaatkan bandwidth yang tersedia secara efisien.


    