import sys
import os
import subprocess
import numpy as np
import shlex

# =========================
# CONFIG
# =========================
RUN_SCRIPT   = 'test-selenium-linux.py'
RANDOM_SEED  = 42
RUN_TIME     = 250          # detik per eksperimen
MM_DELAY     = 40           # millisec — delay jaringan simulasi mahimahi
MM_LINK      = './scaled_traces/report_tram_0001.log'     # kapasitas link mahimahi

# Menggunakan singkatan 'NDN_RL' untuk kestabilan string parsing di Mahimahi Shell
ABR_ALGO     = ['NDN_RL', 'Throughput-Based (HTTP)', 'Buffer-Based (HTTP)']
REPEAT_TIME  = 2
LOG_BASE_DIR = './logs'

# Folder sumber file trace jaringan Mahimahi
TRACE_DIR = './scaled_traces'  

def main():
    np.random.seed(RANDOM_SEED)
    os.makedirs(LOG_BASE_DIR, exist_ok=True)

    # Ambil semua item dari folder dan pastikan hanya mengambil file valid (bukan subfolder)
    if not os.path.exists(TRACE_DIR):
        print(f"[ERROR] Folder trace tidak ditemukan di: {TRACE_DIR}")
        sys.exit(1)
        
    trace_files = sorted([
        f for f in os.listdir(TRACE_DIR) 
        if os.path.isfile(os.path.join(TRACE_DIR, f))
    ])

    if not trace_files:
        print(f"[ERROR] Tidak ada file trace valid di {TRACE_DIR}")
        sys.exit(1)

    with open('./chrome_retry_log.txt', 'w') as log:
        log.write('chrome retry log\n')
        log.flush()

        for rt in range(REPEAT_TIME):
            # Acak urutan ABR setiap repeat
            np.random.shuffle(ABR_ALGO)

            for abr_algo in ABR_ALGO:
                for trace_file in trace_files:
                    trace_path = os.path.join(TRACE_DIR, trace_file)

                    # Subfolder log per kombinasi algo + trace + repeat
                    log_dir = os.path.join(LOG_BASE_DIR, f'{abr_algo}_run{rt}_{trace_file}')
                    os.makedirs(log_dir, exist_ok=True)

                    while True:
                        safe_abr = shlex.quote(abr_algo)
                        safe_trace = shlex.quote(trace_path)
                        safe_log_dir = shlex.quote(log_dir)
                        
                        # Menyusun perintah komando untuk emulasi jaringan Mahimahi
                        cmd = (
                            f'mm-delay {MM_DELAY} '
                            f'mm-link {MM_LINK} {safe_trace} '
                            f'{sys.executable} {RUN_SCRIPT} '
                            f'{safe_abr} {RUN_TIME} {safe_log_dir} {rt}'
                        )

                        print(f"[run {rt}] [{trace_file}] Menjalankan {abr_algo} ...")

                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            shell=True      # Diperlukan oleh Mahimahi
                        )
                        out, err = proc.communicate()

                        out_str = out.decode('utf-8', errors='replace').strip()
                        err_str = err.decode('utf-8', errors='replace').strip()

                        if 'done' in out_str:
                            print(f"[run {rt}] [{trace_file}] {abr_algo} selesai.")
                            break
                        else:
                            # Jika simulasi gagal, catat log lalu lakukan otomatis retry
                            print(f"[run {rt}] [{trace_file}] {abr_algo} gagal, retry...")
                            log.write(f'{abr_algo}_{trace_file}_{rt}\n')
                            log.write(out_str + '\n')
                            if err_str:
                                log.write('STDERR: ' + err_str + '\n')
                            log.write('---\n')
                            log.flush()

if __name__ == '__main__':
    main()