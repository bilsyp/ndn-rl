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
        # ------------------------------------------------------------------
        # 1. HANDLING ACTION & SAFETY NET
        # ------------------------------------------------------------------
        if isinstance(action, (np.ndarray, list)):
            target_idx = int(np.array(action).item())
        else:
            target_idx = int(action)
        
        # Pastikan index selalu dalam batas valid bitrate (misal 0-3)
        target_idx = np.clip(target_idx, 0, NUM_ACTIONS - 1)

        # ------------------------------------------------------------------
        # 2. UNPACKING PREVIOUS STATE
        # ------------------------------------------------------------------
        # Ambil kondisi dari langkah sebelumnya untuk menghitung dampak aksi
        # current_idx di sini adalah bitrate yang sedang berjalan (sebelum diubah)
        buffer, _, current_idx, _, _, prev_cwnd, prev_rtt = self.state
        current_idx = int(current_idx)
        prev_bitrate = self.bitrates[current_idx]

        # ------------------------------------------------------------------
        # 3. UPDATE NETWORK THROUGHPUT (WINDOWING)
        # ------------------------------------------------------------------
        raw_tp = self.trace_manager.get_next_bandwidth()
        self.tp_history.append(raw_tp) # Menggunakan deque(maxlen=20)
        
        # Statistik Jaringan Terkini
        mean_tp = np.mean(self.tp_history)
        volatility = np.std(self.tp_history) / (mean_tp + 1e-6)
        
        # Tambahan: Trend (Apakah jaringan membaik atau memburuk?)
        # Mengukur kemiringan (slope) sederhana dari history
        if len(self.tp_history) > 1:
            trend = (self.tp_history[-1] - self.tp_history[0]) / len(self.tp_history)
        else:
            trend = 0.0

        # ------------------------------------------------------------------
        # 4. SIMULASI METRIK NDN (KAUSALITAS / SEBAB-AKIBAT)
        # ------------------------------------------------------------------
        # Kita hitung load_ratio: Seberapa berat beban bitrate sebelumnya terhadap bandwidth saat ini
        load_ratio = prev_bitrate / (raw_tp + 1e-6)

        # A. RTT Dinamis: Jika load_ratio > 1 (Overload), RTT membengkak (Queueing Delay)
        base_rtt = 200.0 / (raw_tp + 0.5)
        # Tambahkan penalti delay jika bitrate melebihi throughput
        queuing_delay = max(0, (load_ratio - 1.0) * 150.0) 
        new_rtt = np.clip(base_rtt + queuing_delay + np.random.normal(0, 5), 5, 500)

        # B. CWND Dinamis: CWND akan drop jika terjadi kongesti (load_ratio tinggi)
        base_cwnd = raw_tp * 8.0
        if load_ratio > 1.2: # Jaringan mulai sesak
            # Simulasi 'Fast Retransmit' / Congestion Avoidance sederhana
            new_cwnd = np.clip(base_cwnd * 0.4 + np.random.normal(0, 2), 2, 100)
        else:
            new_cwnd = np.clip(base_cwnd + np.random.normal(0, 2), 2, 100)
        # ------------------------------------------------------------------
        # LAYER 2: NDN-AWARE DYNAMIC CONTROLLER (ADAPTIVE GOVERNOR)
        # ------------------------------------------------------------------

        # 1. Tentukan batas buffer dinamis berdasarkan risiko (Volatility & Trend)
        # Jika trend negatif (jaringan turun), kita naikkan syarat buffer
        dynamic_buffer_req = self.LOW_BUFFER_THRESHOLD + (volatility * 10.0)
        if trend < 0:
            dynamic_buffer_req += abs(trend) * 5.0 # Makin cepat turun, makin tinggi syarat buffer

        # Inisialisasi awal
        executed_idx = current_idx 
        is_vetoed = False

        # A. PANIC MODE (Emergency Brake)
        # Jika buffer sangat kritis, langsung turun ke bitrate terendah
        if buffer < self.PANIC_BUFFER_THRESHOLD:
            executed_idx = 0
            is_vetoed = target_idx > 0

        # B. NDN VETO & SMART UPGRADE (Hanya jika tidak Panic)
        else:
            if target_idx > current_idx:
                # 1. Deteksi Kongesti (Causal Feedback)
                # Kita gunakan RTT sebagai sinyal utama kemacetan antrean
                rtt_increase_ratio = new_rtt / (prev_rtt + 1e-6)
                is_congested = rtt_increase_ratio > 1.5 or new_cwnd < self.NDN_CONGESTION_CWND
                
                # 2. Cek Kapasitas (Headroom)
                # Kita cek apakah bitrate tujuan (target_idx) masuk akal dibanding mean_tp
                # Gunakan safety margin yang dipengaruhi oleh volatilitas
                adaptive_margin = self.SAFE_MARGIN * (1.0 + volatility)
                can_support_target = (mean_tp - self.bitrates[target_idx]) > adaptive_margin
                
                # Syarat Upgrade: Tidak macet + Kapasitas ada + Buffer cukup + Trend tidak drop
                if not is_congested and can_support_target and \
                buffer > dynamic_buffer_req and trend >= -0.1:
                    executed_idx = target_idx # Izinkan lompat ke target_idx (lebih agresif)
                else:
                    executed_idx = current_idx # Veto: Tetap di bitrate sekarang
                    is_vetoed = True
                    
            elif target_idx < current_idx:
                # Downgrade: Selalu diizinkan demi keamanan (Exploitation of Safety)
                executed_idx = target_idx
                
            else:
                executed_idx = current_idx

        # Pastikan indeks tetap dalam batas valid
        executed_idx = int(np.clip(executed_idx, 0, NUM_ACTIONS - 1))

       # ------------------------------------------------------------------
        # 2. EKSEKUSI FISIK (Berdasarkan executed_idx yang sudah tervalidasi)
        # ------------------------------------------------------------------
        chosen_bitrate = self.bitrates[executed_idx]

        # Simulasi DASH download (5 detik per segment)
        # Kita beri penalti efisiensi pada bandwidth agar tidak terlalu optimis
        effective_tp = raw_tp * 0.95 
        download_time = (chosen_bitrate * 5.0) / (effective_tp + 0.1)
        
        # Stalling nyata (Detik)
        stalling = max(0, download_time - buffer)
        
        # Update Buffer State (Continuous)
        new_buffer = max(0, buffer - download_time) + 5.0
        new_buffer = min(new_buffer, 30.0)

        # ------------------------------------------------------------------
        # 3. REWARD FUNCTION (SMOOTH & CAUSAL)
        # ------------------------------------------------------------------
         
        # A. Quality Reward (Logarithmic - Memberikan kepuasan yang stabil)
        # Kita gunakan log2 agar kenaikan dari 240p ke 480p terasa signifikan
        reward_quality = np.log2(chosen_bitrate / self.bitrates[0])

        # B. Buffer Safety Reward (Linear Penalty - Menghindari Tembok)
        # Target kita adalah menjaga buffer di area 15-20 detik.
        # Jika di bawah 15 detik, kita kurangi secara linear.
        if new_buffer < 15.0:
            # Penalti makin besar seiring menipisnya buffer (Linear)
            reward_buffer = -1.0 * (15.0 - new_buffer)
        elif new_buffer > 25.0:
            # Bonus kecil untuk buffer yang sangat aman
            reward_buffer = 0.5
        else:
            reward_buffer = 1.0 # Kondisi ideal

        # C. Stalling Penalty (Exponential - Efek Jera Tanpa Mematikan Learning)
        # Daripada langsung -15, kita gunakan fungsi yang membesar seiring durasi
        # Stalling 0.1s tidak terlalu sakit, tapi 2.0s akan sangat sakit.
        penalty_stalling = 0.0
        if stalling > 0:
            penalty_stalling = - (5.0 + 10.0 * np.power(stalling, 1.2))

        # D. Smoothness & Veto Penalty (Belajar Aturan Safety)
        penalty_smoothness = -2.0 * abs(executed_idx - current_idx)
        
        # Penalti Kepatuhan: Jika niat agen (target) ditolak oleh Veto (executed)
        # Ini mengajarkan agen untuk tidak 'meminta' hal yang membahayakan
        penalty_veto = -1.5 if is_vetoed else 0.0

        # TOTAL REWARD
        reward = reward_quality + reward_buffer + penalty_stalling + penalty_smoothness + penalty_veto

        # ------------------------------------------------------------------
        # 4. UPDATE STATE & RETURN
        # ------------------------------------------------------------------
        # Kita tambahkan trend ke dalam state agar agen punya pandangan ke depan
        self.state = np.array(
            [new_buffer, mean_tp, float(executed_idx), trend,
             volatility, new_cwnd, new_rtt],
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
                "stall":   penalty_stalling,
                "smooth":  penalty_smoothness,
                "veto":    penalty_veto
            }
        }
        return self._get_normalized_obs(), reward, done, False, info



def make_env(rank, log_dir, seed=0):
    def _init():
        # 1. Setiap worker membuat manager-nya sendiri
        tm = MahimahiTraceManager(folder_path="../traces_folder/mahimahi_traces")
           
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
    log_dir = "../logs/rl_logs_9bitrate/"
    os.makedirs(log_dir, exist_ok=True)

    tm = MahimahiTraceManager(folder_path="../traces_folder/mahimahi_traces")
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
    
    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=2e-4,     # Konstan dan moderat
        ent_coef=0.03,          # Cukup tinggi untuk eksplorasi, tapi tidak kacau
        n_steps=2048,           # Kembali ke 2048 jika worker sedikit
        batch_size=64,         # Ukuran moderat
        n_epochs=10,            # Biarkan agen belajar lebih dalam dari tiap rollout
        clip_range=0.2,         # Standar agar agen bisa berubah pikiran
        tensorboard_log=log_dir
    )
        
    
    model.learn(total_timesteps=total_steps,
        progress_bar=True,
        tb_log_name="PPO_Parallel_1M_30Traces")
    model.save("hybrid_4bitrate_ndn_model_v8")
    print("✅ Pelatihan selesai. Model disimpan: hybrid_4bitrate_ndn_model_v5")

    # ------------------------------------------------------------------
    # EVALUASI per trace file
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # EVALUASI per trace file (Jalur Khusus)
    # ------------------------------------------------------------------
    print("\n📊 Memulai Evaluasi Akhir...")
    
    # 1. Tutup env paralel untuk membebaskan memori
    env.close()

    # 2. Buat environment tunggal untuk evaluasi agar mendukung 'options'
    # Kita menggunakan DummyVecEnv agar format step tetap konsisten dengan model
    eval_env = DummyVecEnv([lambda: HybridStreamingEnvNDN(tm)])
    
    # --- KONFIGURASI VISUAL ---
    
    res_names = ["240p", "360p", "480p", "720p"]
    res_colors = ["#3498db", "#2ecc71", "#f1c40f", "#e74c3c"] # Biru, Hijau, Kuning, Merah
    for i in range(len(tm.traces)):
        obs = eval_env.reset() 
        eval_env.envs[0].reset(options={"trace_idx": i})
        
        trace_name = tm.traces[i]["name"]
        history = []
        

        for _ in range(120): # Sedikit lebih panjang untuk melihat stabilitas
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)
            
            info_step = infos[0]
            
            # --- FLATTENING DATA INFO ---
            # Kita ambil reward breakdown agar bisa di-plot secara terpisah
            rb = info_step.get("reward_breakdown", {})
            
            history.append({
                "TP":           info_step.get("raw_tp", 0),
                "Volatility":   info_step.get("volatility", 0),
                "Buffer":       info_step.get("buffer", 0),
                "Bitrate":      info_step.get("bitrate", 0),
                "Stalling":     info_step.get("stalling", 0),
                "Vetoed":       info_step.get("is_vetoed", False),
                # Reward Components
                "R_Quality":    rb.get("quality", 0),
                "R_Buffer":     rb.get("buffer", 0),
                "R_Stall":      rb.get("stall", 0),
                "R_Smooth":     rb.get("smooth", 0),
                "R_Veto":       rb.get("veto", 0),
                "Total_R":      rewards[0]
            })
            
            if dones[0]:
                break
        df = pd.DataFrame(history)
        # Tambahkan gridspec_kw untuk mengatur rasio tinggi panel
        fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True, 
                               gridspec_kw={'height_ratios': [1.5, 1, 1, 1]})

        # 1. PANEL 1: THROUGHPUT VS BITRATE (Fokus pada Garis Resolusi)
        ax1 = axes[0]
        # Plot Bandwidth Asli (Pastikan warna terlihat, misal steelblue)
        ax1.plot(df.index, df["TP"], label="Actual Bandwidth", color="steelblue", alpha=0.4, linewidth=1.5)
        # Plot Bitrate yang Dieksekusi (Garis Hitam Tebal)
        ax1.step(df.index, df["Bitrate"], label="Executed Bitrate", color="black", linewidth=2.5, where="post", zorder=10)

        # --- PERBAIKAN GARIS RESOLUSI ---
        for br, label, col in zip(eval_env.envs[0].bitrates, res_names, res_colors):
            # Tingkatkan alpha ke 0.6 agar garis terlihat jelas
            ax1.axhline(y=br, linestyle="--", linewidth=1.2, alpha=0.6, color=col, zorder=1)
            # Pindahkan label ke sisi kanan (right) agar tidak menumpuk di Y-axis
            ax1.text(df.index[-1] * 1.01, br, f"{label}\n({br}M)", fontsize=9, 
                     color=col, fontweight='bold', va='center')

        # Veto Markers (X Merah)
        veto_mask = df["Vetoed"] == True
        if veto_mask.any():
            ax1.scatter(df.index[veto_mask], df["Bitrate"][veto_mask] + 0.15, 
                        color="darkred", marker="x", s=60, label="Veto (Safety Filter)", zorder=11)

        ax1.set_ylabel("Throughput / Bitrate (Mbps)", fontweight='bold')
        ax1.set_title(f"Performance Analysis: {trace_name}", fontsize=16, fontweight='bold', pad=20)
        ax1.legend(loc="upper left", bbox_to_anchor=(0, 1.12), ncol=3, frameon=False)
        ax1.grid(axis='x', alpha=0.2)

        # 2. PANEL 2: BUFFER & STALLING (Gunakan Span untuk Stalling)
        ax2 = axes[1]
        ax2.fill_between(df.index, df["Buffer"], color="forestgreen", alpha=0.15)
        ax2.plot(df.index, df["Buffer"], color="forestgreen", linewidth=2, label="Buffer Level")
        ax2.axhline(y=15.0, color="orange", linestyle="--", linewidth=1.5, label="Safe Threshold (15s)")
        ax2.axhline(y=5.0, color="red", linestyle="-.", linewidth=1.5, label="Panic Threshold (5s)")

        # Tandai Stalling dengan area merah (lebih jelas dibanding garis tipis)
        stall_mask = df["Stalling"] > 0
        if stall_mask.any():
            for idx in df.index[stall_mask]:
                ax2.axvspan(idx-0.5, idx+0.5, color='red', alpha=0.3, label='Stalling' if idx == df.index[stall_mask][0] else "")

        ax2.set_ylabel("Buffer (seconds)", fontweight='bold')
        ax2.set_ylim(0, 35)
        ax2.legend(loc="upper left", fontsize=9)

        # 3. PANEL 3: REWARD COMPONENTS (Analisis Kausalitas)
        ax3 = axes[2]
        ax3.plot(df.index, df["R_Quality"], label="Quality Reward", color="dodgerblue", alpha=0.9)
        ax3.plot(df.index, df["R_Buffer"], label="Buffer Reward", color="orange", alpha=0.9)
        ax3.plot(df.index, df["R_Stall"], label="Stall Penalty", color="crimson", linewidth=2)
        ax3.plot(df.index, df["R_Veto"], label="Veto Penalty", color="purple", linestyle="--")
        ax3.axhline(y=0, color="black", linewidth=1, alpha=0.7)
        ax3.set_ylabel("Reward Components", fontweight='bold')
        ax3.legend(loc="lower left", fontsize=8, ncol=2)

        # 4. PANEL 4: VOLATILITY & TOTAL REWARD
        ax4 = axes[3]
        ax4_t = ax4.twinx()
        p1, = ax4.plot(df.index, df["Volatility"], color="darkmagenta", linewidth=1.5, label="Network Volatility")
        p2, = ax4_t.plot(df.index, df["Total_R"], color="royalblue", linewidth=1.5, label="Total Step Reward")
        
        ax4.set_ylabel("Volatility (CV)", color="darkmagenta", fontweight='bold')
        ax4_t.set_ylabel("Total Reward", color="royalblue", fontweight='bold')
        ax4.set_xlabel("Segment Index", fontweight='bold')
        ax4.legend(handles=[p1, p2], loc="upper left", fontsize=9)
        

        plt.tight_layout()
        out_path = os.path.join(log_dir, f"eval_v3_{trace_name}.png")
        plt.savefig(out_path, dpi=130)
        plt.close()
        print(f"   ✅ Plot Analitik Detail disimpan: {out_path}")
if __name__ == "__main__":
    run_experiment()

    # model terlalu kaku  / terlalu takut untuk naik ke bitrate lebih tinggi karena takut kena veto, padahal sebenarnya jaringan cukup mendukung. Kita perlu memberikan lebih banyak pengalaman kepada agen untuk belajar bahwa beberapa risiko itu layak diambil demi reward jangka panjang yang lebih besar.