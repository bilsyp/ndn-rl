import os
import sys
import time
import shutil
import threading
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# =================================================================
# CONFIGURATION & CONSTANTS
# =================================================================
URL = "https://ndn-memoization-video-client.vercel.app/"
DEFAULT_CHROME_TEMPLATE = './abr_browser_dir/chrome_data_dir'

class ABRExperiment:
    def __init__(self, abr_name, max_runtime, log_dir, repeat_idx):
        self.abr_name = abr_name
        self.max_runtime = int(max_runtime)
        self.log_dir = os.path.abspath(log_dir)
        self.repeat_idx = repeat_idx
        self.pid = str(os.getpid())
        
        # Path profil sementara (kompatibel Windows/Linux)
        self.temp_profile_dir = os.path.join(
            os.environ.get('TEMP', '/tmp'), 
            f'chrome_user_dir_id_{self.pid}'
        )
        
        self.driver = None
        self.watchdog_timer = None

    # ---------------------------------------------------------
    # 1. ENVIRONMENT SETUP
    # ---------------------------------------------------------
    def prepare_environment(self):
        """Menyiapkan folder log dan menyalin profil Chrome bersih."""
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Hapus jika folder profil sementara sudah ada
        if os.path.exists(self.temp_profile_dir):
            shutil.rmtree(self.temp_profile_dir, ignore_errors=True)

        # Salin profil dari template agar tidak ada cache tersisa
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
        """Aksi yang dijalankan jika waktu eksperimen melebihi batas."""
        print(f"\n[TIMEOUT] Batas waktu {self.max_runtime}s tercapai. Memaksa keluar...")
        self.cleanup()
        os._exit(1)

    def start_watchdog(self):
        """Memulai timer pengawas di background."""
        self.watchdog_timer = threading.Timer(self.max_runtime + 60, self._timeout_action) # +60s buffer
        self.watchdog_timer.daemon = True
        self.watchdog_timer.start()

    # ---------------------------------------------------------
    # 3. BROWSER INTERACTION
    # ---------------------------------------------------------
    def launch_browser(self):
        """Inisialisasi Chrome dengan opsi tertentu."""
        options = webdriver.ChromeOptions()
        
        # Setting agar log otomatis terunduh ke direktori eksperimen
        prefs = {
            "download.default_directory": self.log_dir,
            "download.prompt_for_download": False,
            "profile.default_content_setting_values.automatic_downloads": 1
        }
        options.add_experimental_option("prefs", prefs)
        options.add_argument(f"--user-data-dir={self.temp_profile_dir}")
        options.add_argument("--window-size=800,600")
        
        # Untuk Linux/Server (Opsional):
        # options.add_argument("--headless") 

        self.driver = webdriver.Chrome(options=options)
        print("[BROWSER] Chrome diluncurkan.")

    def run_simulation(self):
        """Menjalankan alur eksperimen di web."""
        try:
            self.driver.get(URL)
            
            # Tunggu dropdown muncul (max 20 detik)
            # wait = WebDriverWait(self.driver, 20)
            time.sleep(10) # Delay Connection
            select_element = self.driver.find_element(By.ID,"selectAbr")
            
            # Pilih Algoritma ABR
            abr_dropdown = Select(select_element)
            abr_dropdown.select_by_visible_text(self.abr_name)
            
            print(f"[RUNNING] ABR: {self.abr_name} | Repeat: {self.repeat_idx}")
            print(f"[WAIT] Menunggu durasi eksperimen: {self.max_runtime} detik...")
            
            # Tunggu hingga durasi video selesai
            time.sleep(self.max_runtime)
            
            return True
        except Exception as e:
            print(f"[ERROR] Terjadi kesalahan saat simulasi: {e}")
            return False

    def download_logs(self):
        """Memicu klik tombol download di halaman web."""
        print("[INFO] Mengunduh log...")
        try:
           # 1. Tentukan daftar tombol berdasarkan logika algoritma
            if self.abr_name == "NDN_RL (Named Data Networking)":
                buttons = {
                    "Memo Log": "memo_download",
                    "Latency Log": "latency_download",
                    "QoE Log": "qoe_download"  # Menambahkan qoe_download sesuai permintaan
                }
            else:
                # Untuk algoritma selain NDN_RL, hanya download QoE saja
                buttons = {
                    "QoE Log": "qoe_download"
                }
            
            for name, btn_id in buttons.items():
                btn = self.driver.find_element(By.ID, btn_id)
                btn.click()
                print(f"  - {name} diklik.")
                time.sleep(1.5) # Jeda singkat antar download
                
            print("[SUCCESS] Semua log telah diproses.")
        except NoSuchElementException:
            print("[ERROR] Tombol download tidak ditemukan di halaman.")
        except Exception as e:
            print(f"[ERROR] Gagal mengunduh log: {e}")

    # ---------------------------------------------------------
    # 4. CLEANUP
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
    if len(sys.argv) < 5:
        print("Usage: python test-selenium.py <abr_name> <max_runtime> <log_dir> <repeat_index>")
        sys.exit(1)

    # Inisialisasi Objek
    exp = ABRExperiment(
        abr_name=sys.argv[1],
        max_runtime=sys.argv[2],
        log_dir=sys.argv[3],
        repeat_idx=sys.argv[4]
    )

    try:
        exp.prepare_environment()
        exp.start_watchdog()
        exp.launch_browser()
        
        if exp.run_simulation():
            exp.download_logs()
            print("done") # Sinyal untuk script orchestrator
            
    finally:
        exp.cleanup()
        print("[EXIT] Program selesai.")