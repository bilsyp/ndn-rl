import sys
import os
import subprocess
import numpy as np

# =========================
# CONFIG
# =========================
RUN_SCRIPT   = 'test-selenium-linux.py'
RANDOM_SEED  = 42
RUN_TIME     = 250          # detik per eksperimen
MM_DELAY     = 40           # millisec — delay jaringan simulasi mahimahi
MM_LINK      = '6mbps'     # kapasitas link mahimahi
ABR_ALGO     = ['NDN_RL (Named Data Networking)', 'Throughput-Based (HTTP)', 'Buffer-Based (HTTP)']
REPEAT_TIME  = 2
LOG_BASE_DIR = '../logs'
TRACE_DIR    = './traces/report_tram_*.log'  # folder berisi file-file trace jaringan mahimahi


def main():
    np.random.seed(RANDOM_SEED)

    os.makedirs(LOG_BASE_DIR, exist_ok=True)

    # Ambil semua file trace dari folder
    trace_files = sorted(os.listdir(TRACE_DIR))
    if not trace_files:
        print(f"[ERROR] Tidak ada file trace di {TRACE_DIR}")
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
                        # Mahimahi membungkus perintah selenium di dalam jaringan simulasi:
                        # mm-delay <delay_ms> mm-link <kapasitas> <trace_file> <perintah>
                        cmd = (
                            f'mm-delay {MM_DELAY} '
                            f'mm-link {MM_LINK} {trace_path} '
                            f'{sys.executable} {RUN_SCRIPT} '
                            f'"{abr_algo}" {RUN_TIME} {log_dir} {rt}'
                        )

                        print(f"[run {rt}] [{trace_file}] Menjalankan {abr_algo} ...")

                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            shell=True      # shell=True diperlukan oleh mahimahi
                        )
                        out, err = proc.communicate()

                        out_str = out.decode('utf-8', errors='replace').strip()
                        err_str = err.decode('utf-8', errors='replace').strip()

                        if 'done' in out_str:
                            print(f"[run {rt}] [{trace_file}] {abr_algo} selesai.")
                            break
                        else:
                            # Gagal → catat ke log lalu retry
                            print(f"[run {rt}] [{trace_file}] {abr_algo} gagal, retry...")
                            log.write(f'{abr_algo}_{trace_file}_{rt}\n')
                            log.write(out_str + '\n')
                            if err_str:
                                log.write('STDERR: ' + err_str + '\n')
                            log.write('---\n')
                            log.flush()


if __name__ == '__main__':
    main()