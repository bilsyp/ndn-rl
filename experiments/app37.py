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
        # KOREKSI: Pemotongan unpacking state dari 7 elemen menjadi 6 elemen
        buffer, mean_tp, last_exec, _, volatility, cwnd = self.state

        norm_buffer = np.clip(buffer / 30.0, 0.0, 1.0)
        norm_tp     = np.clip(mean_tp / SCALE_TARGET, 0.0, 1.0)  
        norm_exec   = float(last_exec) / float(NUM_ACTIONS - 1)

        safety      = max(0, buffer - self.LOW_BUFFER_THRESHOLD)
        norm_safety = np.clip(safety / 20.0, 0.0, 1.0)
        norm_vol    = np.clip(volatility, 0.0, 1.0)
        norm_cwnd   = np.clip(cwnd / 100.0, 0.0, 1.0)

        # KOREKSI: Menghapus elemen norm_rtt dari kembalian array observasi
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
        if buffer < self.PANIC_BUFFER_THRESHOLD:
            executed_idx = 0
            is_vetoed = target_idx > 0
            if is_vetoed:
                violated_gates["buffer"] = True # Otomatis melanggar pertahanan buffer

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
                "veto":    penalty_gate_break
            }
        }
        return self._get_normalized_obs(), reward, done, False, info


def make_env(rank, log_dir, seed=0):
    def _init():
        tm = MahimahiTraceManager(folder_path="../traces_folder/mahimahi_traces")
        env = HybridStreamingEnvNDN(tm)
        monitor_path = os.path.join(log_dir, str(rank))
        env = Monitor(env, monitor_path)
        return env
    return _init


def run_experiment():
    log_dir = "../logs/rl_logs_14bitrate/"
    os.makedirs(log_dir, exist_ok=True)

    tm = MahimahiTraceManager(folder_path="../traces_folder/mahimahi_traces")
    if not tm.traces:
        return

    num_cpu = 4
    total_steps = 1000000
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
        
    model.learn(total_timesteps=total_steps, progress_bar=True, tb_log_name="PPO_Parallel_1M_30Traces")
    model.save("../models/hybrid_14bitrate_ndn_model_v14")
    print("✅ Pelatihan selesai. Model disimpan: hybrid_14bitrate_ndn_model_v14.zip")

    print("\n📊 Memulai Evaluasi Akhir...")
    env.close()

    eval_env = DummyVecEnv([lambda: HybridStreamingEnvNDN(tm)])
    
    res_names = ["240p", "360p", "480p", "720p"]
    res_colors = ["#3498db", "#2ecc71", "#f1c40f", "#e74c3c"] 
    
    for i in range(len(tm.traces)):
        obs = eval_env.reset() 
        eval_env.envs[0].reset(options={"trace_idx": i})
        
        trace_name = tm.traces[i]["name"]
        history = []
        
        for _ in range(120): 
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)
            
            info_step = infos[0]
            rb = info_step.get("reward_breakdown", {})
            
            history.append({
                "TP":           info_step.get("raw_tp", 0),
                "Volatility":   info_step.get("volatility", 0),
                "Buffer":       info_step.get("buffer", 0),
                "Bitrate":      info_step.get("bitrate", 0),
                "Stalling":     info_step.get("stalling", 0),
                "Vetoed":       info_step.get("is_vetoed", False),
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
        fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True, 
                               gridspec_kw={'height_ratios': [1.5, 1, 1, 1]})

        # 1. PANEL 1: THROUGHPUT VS BITRATE
        ax1 = axes[0]
        ax1.plot(df.index, df["TP"], label="Actual Bandwidth", color="steelblue", alpha=0.4, linewidth=1.5)
        ax1.step(df.index, df["Bitrate"], label="Executed Bitrate", color="black", linewidth=2.5, where="post", zorder=10)

        for br, label, col in zip(eval_env.envs[0].bitrates, res_names, res_colors):
            ax1.axhline(y=br, linestyle="--", linewidth=1.2, alpha=0.6, color=col, zorder=1)
            ax1.text(df.index[-1] * 1.01, br, f"{label}\n({br}M)", fontsize=9, color=col, fontweight='bold', va='center')

        veto_mask = df["Vetoed"] == True
        if veto_mask.any():
            ax1.scatter(df.index[veto_mask], df["Bitrate"][veto_mask] + 0.15, 
                        color="darkred", marker="x", s=60, label="Veto (Safety Filter)", zorder=11)

        ax1.set_ylabel("Throughput / Bitrate (Mbps)", fontweight='bold')
        ax1.set_title(f"Performance Analysis (Refactored): {trace_name}", fontsize=16, fontweight='bold', pad=20)
        ax1.legend(loc="upper left", bbox_to_anchor=(0, 1.12), ncol=3, frameon=False)
        ax1.grid(axis='x', alpha=0.2)

        # 2. PANEL 2: BUFFER & STALLING
        ax2 = axes[1]
        ax2.fill_between(df.index, df["Buffer"], color="forestgreen", alpha=0.15)
        ax2.plot(df.index, df["Buffer"], color="forestgreen", linewidth=2, label="Buffer Level")
        ax2.axhline(y=15.0, color="orange", linestyle="--", linewidth=1.5, label="Safe Threshold (15s)")
        ax2.axhline(y=5.0, color="red", linestyle="-.", linewidth=1.5, label="Panic Threshold (5s)")

        stall_mask = df["Stalling"] > 0
        if stall_mask.any():
            for idx in df.index[stall_mask]:
                ax2.axvspan(idx-0.5, idx+0.5, color='red', alpha=0.3, label='Stalling' if idx == df.index[stall_mask][0] else "")

        ax2.set_ylabel("Buffer (seconds)", fontweight='bold')
        ax2.set_ylim(0, 35)
        ax2.legend(loc="upper left", fontsize=9)

        # 3. PANEL 3: REWARD COMPONENTS
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
        out_path = os.path.join(log_dir, f"eval_v4_clean_{trace_name}.png")
        plt.savefig(out_path, dpi=130)
        plt.close()
        print(f"   ✅ Plot Analitik Bersih disimpan: {out_path}")


if __name__ == "__main__":
    run_experiment()

    # catatan:


    