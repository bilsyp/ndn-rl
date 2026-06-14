
import os
import sys
import time
import threading
import glob
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# =================================================================
# CONFIGURATION & CONSTANTS
# =================================================================
URL = "http://192.168.1.14:3000"

# =================================================================
# QoE CALCULATION
# =================================================================
def calculate_avg_qoe_pandas(file_path):
    """Menghitung rata-rata QoE dari file CSV yang baru didownload."""
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        qoe_scores = pd.to_numeric(df['qoe_score'], errors='coerce')
        qoe_scores = qoe_scores.fillna(0)
        avg_qoe = qoe_scores.mean()
        return avg_qoe
    except Exception as e:
        print(f"[QoE ERROR] Gagal menghitung QoE dari {file_path}: {e}")
        return None


class ABRExperiment:
    def __init__(self, abr_name, max_runtime, log_dir, repeat_idx, trace_file):
        self.abr_name    = abr_name
        self.max_runtime = int(max_runtime)
        self.log_dir     = os.path.abspath(log_dir)
        self.repeat_idx  = repeat_idx
        self.trace_file  = trace_file

        self._pw            = None
        self.browser        = None
        self.context        = None
        self.page           = None
        self.watchdog_timer = None

    # ---------------------------------------------------------
    # 1. ENVIRONMENT SETUP
    # ---------------------------------------------------------
    def prepare_environment(self):
        """Menyiapkan folder log. Profil temp tidak diperlukan di Playwright."""
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[SETUP] Folder log siap: {self.log_dir}")

    # ---------------------------------------------------------
    # 2. WATCHDOG TIMER (Timeout Handler)
    # ---------------------------------------------------------
    def _timeout_action(self):
        print(f"\n[TIMEOUT] Batas waktu {self.max_runtime}s tercapai. Memaksa keluar...")
        self.cleanup()
        os._exit(1)

    def start_watchdog(self):
        self.watchdog_timer = threading.Timer(self.max_runtime + 90, self._timeout_action)
        self.watchdog_timer.daemon = True
        self.watchdog_timer.start()

    # ---------------------------------------------------------
    # 3. BROWSER INTERACTION
    # ---------------------------------------------------------
    def launch_browser(self):
        """Inisialisasi Chromium via Playwright untuk Linux & Mahimahi."""
        self._pw = sync_playwright().start()

        self.browser = self._pw.chromium.launch(
            headless=False,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ]
        )

        # Context menggantikan profil Chrome — isolasi per-run, support download
        self.context = self.browser.new_context(
            accept_downloads=True,
            viewport={"width": 800, "height": 600},
        )

        self.page = self.context.new_page()
        print("[BROWSER] Chromium (Playwright) diluncurkan.")

    def run_simulation(self):
        """Menjalankan alur eksperimen di web."""
        try:
            print(f"[BROWSER] Membuka URL: {URL}")
            self.page.goto(URL, timeout=60_000)
            time.sleep(10)

            print("[WAIT] Menunggu elemen dropdown 'selectAbr' siap...")
            # Playwright auto-wait — tidak perlu time.sleep(10)
            self.page.wait_for_selector("#selectAbr", timeout=45_000)

            try:
                self.page.select_option("#selectAbr", value=self.abr_name)
                print(f"[SUCCESS] Berhasil memilih algoritma berdasarkan VALUE: {self.abr_name}")
            except Exception:
                self.page.select_option("#selectAbr", label=self.abr_name)
                print(f"[SUCCESS] Berhasil memilih algoritma berdasarkan TEXT: {self.abr_name}")

            print(f"[RUNNING] ABR: {self.abr_name} | Repeat: {self.repeat_idx}")
            print(f"[WAIT] Menunggu durasi eksperimen: {self.max_runtime} detik...")

            time.sleep(self.max_runtime)
            time.sleep(3)
            return True

        except PlaywrightTimeout:
            print("[ERROR] Timeout: Halaman/elemen web terlalu lama dimuat akibat jaringan lambat.")
            return False
        except Exception as e:
            print(f"[ERROR] Terjadi kesalahan saat simulasi: {e}")
            return False

    def download_logs(self):
        """Memicu klik tombol download, simpan file, lalu hitung QoE."""
        print("[INFO] Mengunduh log...")
        try:
            if "NDN_RL" in self.abr_name:
                buttons = {
                    "Memo Log":    "memo_download",
                    "Latency Log": "latency_download",
                    "QoE Log":     "qoe_download",
                }
            else:
                buttons = {
                    "QoE Log": "qoe_download",
                }

            for name, btn_id in buttons.items():
                # expect_download() menangkap event download — tidak perlu sleep
                with self.page.expect_download(timeout=20_000) as dl_info:
                    self.page.click(f"#{btn_id}", timeout=20_000)

                download = dl_info.value
                dest = os.path.join(self.log_dir, download.suggested_filename)
                download.save_as(dest)
                print(f"  - {name} tersimpan: {dest}")

            print("[SUCCESS] Semua log telah diproses.")
            self._process_qoe()

        except PlaywrightTimeout:
            print("[ERROR] Tombol download tidak dapat diklik (Timeout).")
        except Exception as e:
            print(f"[ERROR] Gagal mengunduh log: {e}")

    # ---------------------------------------------------------
    # 4. QoE PROCESSING & ACCUMULATION
    # ---------------------------------------------------------
    def _process_qoe(self):
        qoe_files = glob.glob(os.path.join(self.log_dir, 'QoE_Report_*.csv'))

        if not qoe_files:
            print("[QoE WARN] Tidak ada file QoE ditemukan, lewati perhitungan.")
            return

        qoe_file = max(qoe_files, key=os.path.getmtime)
        avg_qoe = calculate_avg_qoe_pandas(qoe_file)
        if avg_qoe is None:
            return

        label = f"{self.abr_name}_run{self.repeat_idx}_{self.trace_file}"
        safe_algo_name = self.abr_name.replace('/', '-').replace(' ', '_')

        logs_root = os.path.dirname(self.log_dir)
        os.makedirs(logs_root, exist_ok=True)

        accum_file = os.path.join(logs_root, f'{safe_algo_name}_qoe_results.txt')

        with open(accum_file, 'a') as f:
            f.write(f"{label} : {avg_qoe:.2f}\n")

        print(f"[QoE] Hasil disimpan → {accum_file}")

    # ---------------------------------------------------------
    # 5. CLEANUP
    # ---------------------------------------------------------
    def cleanup(self):
        """Membersihkan resource (page, context, browser, playwright, timer)."""
        if self.watchdog_timer:
            self.watchdog_timer.cancel()

        try:
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self._pw:
                self._pw.stop()
            print("[CLEANUP] Browser ditutup.")
        except Exception:
            pass


# =================================================================
# MAIN EXECUTION
# =================================================================
if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python test-playwright-linux.py <abr_name> <max_runtime> <log_dir> <repeat_index> <trace_file>")
        sys.exit(1)

    exp = ABRExperiment(
        abr_name    = sys.argv[1],
        max_runtime = sys.argv[2],
        log_dir     = sys.argv[3],
        repeat_idx  = sys.argv[4],
        trace_file  = sys.argv[5],
    )

    try:
        exp.prepare_environment()
        exp.start_watchdog()
        exp.launch_browser()

        if exp.run_simulation():
            exp.download_logs()
            print("done")

    finally:
        exp.cleanup()
        print("[EXIT] Program selesai.")
