### ---

> [!NOTE]
>
> **Otomasi Penuh:** Seluruh proses pengujian dilakukan secara otomatis. Pemilihan skenario hingga interaksi dengan elemen web telah dikonfigurasi melalui skrip **Selenium**. Penguji tidak perlu melakukan interaksi manual tambahan pada browser agar tidak mengganggu jalannya eksperimen.

**File: automation/README.md**

# Automation Module

Folder ini berisi skrip otomasi eksperimen menggunakan **Selenium** untuk pengujian browser dan **Mahimahi** sebagai simulator kondisi jaringan (_network scenario_). Eksperimen dijalankan berdasarkan _network traces_ yang telah diproses sebelumnya.

## 📂 Struktur Folder

- **`run-exp/`**
  Direktori utama yang menyimpan seluruh hasil keluaran (_output_) dari setiap sesi eksperimen.

- **`traces/`**
  Folder lokal yang berisi file _trace_ jaringan (format Mahimahi) yang sudah di-scale.

  > **Catatan:** Pastikan file dari `../traces_folder/scaled_traces/` sudah disinkronkan ke folder ini jika diperlukan.

- **`scaled_traces/`**
  Berisi logs _trace_ dengan berbagai tipe kondisi mobilitas:
  - `car` (Mobil)
  - `berjalan` (Foot)
  - `tram`
  - `bus`
  - `kereta` (Train)

> [!IMPORTANT]
> **Saran Eksperimen:** Pilih salah satu kategori saja dari daftar di atas (misal: `car`) untuk tahap pengujian awal agar data lebih konsisten sebelum mencoba semua kategori.

## 🚀 Cara Menjalankan

Cukup jalankan skrip utama menggunakan Python:

```bash
python run_experiment-linux.py
```

## **⚙️ Konfigurasi Eksperimen**

Sebelum menjalankan, perhatikan variabel konfigurasi di dalam run_experiment-linux.py. Berikut adalah pengaturan bawaan yang digunakan:

| Variabel     | Nilai Default            | Deskripsi                                                                       |
| :----------- | :----------------------- | :------------------------------------------------------------------------------ |
| RUN_SCRIPT   | 'test-selenium-linux.py' | Skrip Selenium yang akan dieksekusi oleh runner.                                |
| RANDOM_SEED  | 42                       | Menjamin hasil eksperimen dapat direplikasi (reproducible).                     |
| RUN_TIME     | 250                      | Durasi setiap sesi eksperimen dalam satuan **detik**.                           |
| MM_DELAY     | 40                       | Delay jaringan simulasi Mahimahi dalam satuan **ms**.                           |
| MM_LINK      | '12mbps'                 | Kapasitas bandwidth uplink pada simulasi Mahimahi.(tapi ini kagak ngaruh sih :) |
| REPEAT_TIME  | 2                        | Jumlah pengulangan untuk setiap skenario eksperimen.                            |
| LOG_BASE_DIR | '../logs'                | Lokasi penyimpanan file log hasil eksperimen.                                   |
| TRACE_DIR    | './traces/'              | Folder sumber file trace jaringan Mahimahi.                                     |

### **Algoritma ABR yang Diuji:**

Eksperimen ini akan membandingkan performa dari tiga algoritma berikut:

1. **NDN_RL** (Named Data Networking \- Reinforcement Learning)
2. **Throughput-Based** (HTTP)
3. **Buffer-Based** (HTTP)

---

---

proses pengujian ini dilakukan secara automation / otomatis semua pemilihan , interaksi dengan web sudah diseeting pada script selenium jadi untuk penguji tidak perlu ada interaksi tambahan yang tidak relevan

## 🛠️ Catatan Penting & Penanganan Masalah

- **Verifikasi Output:** Setelah pengujian selesai, pastikan folder `logs/` telah terisi dengan data yang sesuai untuk memastikan eksperimen berjalan sempurna.
- **Pelaporan Masalah:** Jika ditemukan _error_ atau _bug_ selama proses pengujian, harap catat detail kejadiannya dan laporkan melalui **GitHub Issues** pada repositori ini.
- **Persyaratan Sistem:** Modul ini sangat bergantung pada keberadaan _binary_ **Mahimahi** di dalam _path_ sistem Anda.
- **Izin Eksekusi:** Pastikan ada koneksi internet (untuk Selenium & Mahimahi) dan berikan izin eksekusi (_executable permission_) pada skrip sebelum dijalankan.

**Peringatan:** Pastikan environment Mahimahi sudah terinstal dan terkonfigurasi dengan benar di sistem operasi (Linux direkomendasikan) sebelum menjalankan eksperimen.
