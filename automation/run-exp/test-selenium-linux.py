import os
import sys
import time
import shutil
import threading
import glob
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# =================================================================
# CONFIGURATION & CONSTANTS
# =================================================================
URL = "https://ndn-memoization-video-client.vercel.app/"
DEFAULT_CHROME_TEMPLATE = './abr_browser_dir/chrome_data_dir'

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
        self.trace_file  = trace_file      # [NEW] nama file trace untuk label akumulasi
        self.pid         = str(os.getpid())

        # Path profil sementara (kompatibel Windows/Linux)
        self.temp_profile_dir = os.path.join(
            os.environ.get('TMPDIR', '/tmp'),
            f'chrome_user_dir_id_{self.pid}'
        )

        self.driver        = None
        self.watchdog_timer = None

    # ---------------------------------------------------------
    # 1. ENVIRONMENT SETUP
    # ---------------------------------------------------------
    def prepare_environment(self):
        """Menyiapkan folder log dan menyalin profil Chrome bersih."""
        os.makedirs(self.log_dir, exist_ok=True)

        if os.path.exists(self.temp_profile_dir):
            shutil.rmtree(self.temp_profile_dir, ignore_errors=True)

        if os.path.exists(DEFAULT_CHROME_TEMPLATE):
            shutil.copytree(DEFAULT_CHROME_TEMPLATE, self.temp_profile_dir)
            print(f"[SETUP] Profil disalin ke: {self.temp_profile_dir}")
        else:
            os.makedirs(self.temp_profile_dir, exist_ok=True)
            print("[WARN] Template profil tidak ada, menggunakan profil kosong.")

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
        """Inisialisasi Chrome dengan opsi stabilitas tinggi untuk Linux & Mahimahi."""
        options = webdriver.ChromeOptions()

        prefs = {
            "download.default_directory": self.log_dir,
            "download.prompt_for_download": False,
            "profile.default_content_setting_values.automatic_downloads": 1
        }
        options.add_experimental_option("prefs", prefs)
        options.add_argument(f"--user-data-dir={self.temp_profile_dir}")
        options.add_argument("--window-size=800,600")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        print("[BROWSER] Chrome diluncurkan dengan konfigurasi stabilitas jaringan.")

    def run_simulation(self):
        """Menjalankan alur eksperimen di web dengan penanganan Explicit Wait."""
        try:
            print(f"[BROWSER] Membuka URL: {URL}")
            self.driver.get(URL)

            wait = WebDriverWait(self.driver, 45)
            print("[WAIT] Menunggu elemen dropdown 'selectAbr' siap...")
            time.sleep(10)
            select_element = wait.until(EC.presence_of_element_located((By.ID, "selectAbr")))

            abr_dropdown = Select(select_element)

            try:
                abr_dropdown.select_by_value(self.abr_name)
                print(f"[SUCCESS] Berhasil memilih algoritma berdasarkan VALUE: {self.abr_name}")
            except NoSuchElementException:
                abr_dropdown.select_by_visible_text(self.abr_name)
                print(f"[SUCCESS] Berhasil memilih algoritma berdasarkan TEXT: {self.abr_name}")

            print(f"[RUNNING] ABR: {self.abr_name} | Repeat: {self.repeat_idx}")
            print(f"[WAIT] Menunggu durasi eksperimen: {self.max_runtime} detik...")

            time.sleep(self.max_runtime)
            time.sleep(3)
            return True

        except TimeoutException:
            print("[ERROR] Timeout: Halaman/elemen web terlalu lama dimuat akibat jaringan lambat.")
            return False
        except Exception as e:
            print(f"[ERROR] Terjadi kesalahan saat simulasi: {e}")
            return False

    def download_logs(self):
        """Memicu klik tombol download, lalu hitung QoE dan akumulasi hasilnya."""
        print("[INFO] Mengunduh log...")
        try:
            if "NDN_RL" in self.abr_name:
                buttons = {
                    "Memo Log":    "memo_download",
                    "Latency Log": "latency_download",
                    "QoE Log":     "qoe_download"
                }
            else:
                buttons = {
                    "QoE Log": "qoe_download"
                }

            wait = WebDriverWait(self.driver, 20)

            for name, btn_id in buttons.items():
                btn = wait.until(EC.element_to_be_clickable((By.ID, btn_id)))
                btn.click()
                print(f"  - {name} berhasil diklik.")
                time.sleep(5)

            print("[SUCCESS] Semua log telah diproses.")

            # [NEW] Setelah download selesai, cari file QoE dan hitung rata-ratanya
            self._process_qoe()

        except TimeoutException:
            print("[ERROR] Tombol download tidak dapat diklik (Timeout).")
        except NoSuchElementException:
            print("[ERROR] Tombol download tidak ditemukan di halaman.")
        except Exception as e:
            print(f"[ERROR] Gagal mengunduh log: {e}")

    # ---------------------------------------------------------
    # [NEW] 4. QoE PROCESSING & ACCUMULATION
    # ---------------------------------------------------------
    def _process_qoe(self):
        """
        Cari file QoE yang baru didownload, hitung rata-rata,
        lalu tambahkan ke file akumulasi per algoritma di LOG_BASE_DIR.

        Format baris hasil:
          NDN_RL (Named Data Networking)_run0_report_bicycle_0001.log : -3.08
        """
        # Cari file CSV QoE di log_dir — pola: *qoe*.csv
        qoe_files = glob.glob(os.path.join(self.log_dir, '*qoe*.csv'))

        if not qoe_files:
            print("[QoE WARN] Tidak ada file QoE ditemukan, lewati perhitungan.")
            return

        # Ambil file terbaru jika ada lebih dari satu
        qoe_file = max(qoe_files, key=os.path.getmtime)
        print(f"[QoE] Menghitung rata-rata dari: {os.path.basename(qoe_file)}")

        avg_qoe = calculate_avg_qoe_pandas(qoe_file)
        if avg_qoe is None:
            print("[QoE WARN] Perhitungan QoE gagal, baris tidak ditambahkan.")
            return

        # [NEW] Bentuk label sesuai format yang diinginkan
        # Contoh: "NDN_RL (Named Data Networking)_run0_report_bicycle_0001.log"
        label = f"{self.abr_name}_run{self.repeat_idx}_{self.trace_file}"

        # [NEW] Nama file akumulasi per algoritma di root ./logs/
        # Karakter "/" dan spasi di nama algo diganti agar aman sebagai nama file
        safe_algo_name = self.abr_name.replace('/', '-').replace(' ', '_')
        # LOG_BASE_DIR = direktori induk dari self.log_dir (./logs/)
        logs_root = os.path.dirname(self.log_dir) if "NDN_RL" in self.abr_name else self.log_dir
        accum_file = os.path.join(logs_root, f'{safe_algo_name}_qoe_results.txt')

        # Tambahkan baris ke file akumulasi (append)
        with open(accum_file, 'a') as f:
            f.write(f"{label} : {avg_qoe:.2f}\n")

        print(f"[QoE] Hasil disimpan → {accum_file}")
        print(f"[QoE] {label} : {avg_qoe:.2f}")

    # ---------------------------------------------------------
    # 5. CLEANUP
    # ---------------------------------------------------------
    def cleanup(self):
        """Membersihkan resource (driver, timer, profil sementara)."""
        if self.watchdog_timer:
            self.watchdog_timer.cancel()

        if self.driver:
            try:
                self.driver.quit()
                print("[CLEANUP] Browser ditutup.")
            except:
                pass

        if os.path.exists(self.temp_profile_dir):
            shutil.rmtree(self.temp_profile_dir, ignore_errors=True)
            print(f"[CLEANUP] Profil sementara dihapus.")

# =================================================================
# MAIN EXECUTION
# =================================================================
if __name__ == "__main__":
    # [CHANGED] Argumen ke-6 (trace_file) sekarang wajib
    if len(sys.argv) < 6:
        print("Usage: python test-selenium-linux.py <abr_name> <max_runtime> <log_dir> <repeat_index> <trace_file>")
        sys.exit(1)

    exp = ABRExperiment(
        abr_name    = sys.argv[1],
        max_runtime = sys.argv[2],
        log_dir     = sys.argv[3],
        repeat_idx  = sys.argv[4],
        trace_file  = sys.argv[5],   # [NEW]
    )

    try:
        exp.prepare_environment()
        exp.start_watchdog()
        exp.launch_browser()

        if exp.run_simulation():
            exp.download_logs()
            print("done")  # Sinyal sukses untuk script orchestrator

    finally:
        exp.cleanup()
        print("[EXIT] Program selesai.")