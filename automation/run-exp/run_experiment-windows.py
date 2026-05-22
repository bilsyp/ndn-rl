import sys
import os
import subprocess
import numpy as np

# =========================
# CONFIG
# =========================
RUN_SCRIPT   = 'test-selenium-windows.py'
RANDOM_SEED  = 42
RUN_TIME     = 250          # detik per eksperimen
ABR_ALGO     = ['RL (HTTP)','NDN_RL (Named Data Networking)', 'Throughput-Based (HTTP)', 'Buffer-Based (HTTP)']
REPEAT_TIME  = 2
LOG_BASE_DIR = '../logs'     # folder log utama; subfolder dibuat otomatis per run


def main():
    np.random.seed(RANDOM_SEED)

    os.makedirs(LOG_BASE_DIR, exist_ok=True)

    with open('./chrome_retry_log.txt', 'w') as log:
        log.write('chrome retry log\n')
        log.flush()

        for rt in range(REPEAT_TIME):
            # Acak urutan ABR setiap repeat
            np.random.shuffle(ABR_ALGO)

            for abr_algo in ABR_ALGO:
                # Buat subfolder log khusus untuk run ini
                log_dir = os.path.join(LOG_BASE_DIR, f'{abr_algo}_run{rt}')
                os.makedirs(log_dir, exist_ok=True)

                while True:
                    # Format: python test-selenium.py <abr> <runtime> <log_dir> <repeat_index>
                    cmd = [
                        sys.executable,     # path python yang sedang dipakai
                        RUN_SCRIPT,
                        abr_algo,
                        str(RUN_TIME),
                        log_dir,
                        str(rt)
                    ]

                    print(f"[run {rt}] Menjalankan {abr_algo} ...")

                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    out, err = proc.communicate()

                    # Decode bytes → string (Python 3)
                    out_str = out.decode('utf-8', errors='replace').strip()
                    err_str = err.decode('utf-8', errors='replace').strip()

                    if 'done' in out_str:
                        print(f"[run {rt}] {abr_algo} selesai.")
                        break
                    else:
                        # Gagal → catat ke log lalu retry
                        print(f"[run {rt}] {abr_algo} gagal, retry...")
                        log.write(f'{abr_algo}_{rt}\n')
                        log.write(out_str + '\n')
                        if err_str:
                            log.write('STDERR: ' + err_str + '\n')
                        log.write('---\n')
                        log.flush()


if __name__ == '__main__':
    main()