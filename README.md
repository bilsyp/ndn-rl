# Project Name: Network Trace Training & Automation

Proyek ini bertujuan untuk melakukan pemrosesan data network trace (Belgium Dataset), melakukan konversi ke format Mahimahi, scaling bandwidth, dan menjalankan eksperimen otomatis menggunakan Selenium.

## 🛠 Prasyarat (Prerequisites)

- **Python:** v3.11.9
- **OS:** Windows / Linux (Ubuntu direkomendasikan untuk Mahimahi)
- **Tools:** Chrome/Chromedriver (untuk automation)

## 🚀 Instalasi

1. Clone repository ini.
2. Pastikan Anda berada di direktori utama proyek.
3. Buat dan aktifkan virtual environment:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux
   source venv/bin/activate
   ```

### ---

**Penjelasan Detil Struktur Folder**

| Folder / File      | Fungsi & Alasan Penempatan                                                                                                                                                                    |
| :----------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **automation/**    | Memisahkan skrip penggerak browser. Folder ini berisi "cara menguji model di web dengan mahimahi trace". Dipisahkan agar tidak mengganggu folder utama yang berisi logika inti model.         |
| **experiments/**   | Tempat untuk "mencoba-coba". dan membuat otak backlip 50x.                                                                                                                                    |
| **logs/**          | Berisi catatan proses pelatihan Model.                                                                                                                                                        |
| **models/**        | "Gudang" hasil jadi. Setelah proses training di experiments selesai, file model yang sudah matang disimpan di sini agar siap dipanggil oleh folder automation tanpa perlu men-training ulang. |
| **traces_folder/** | **Pusat Data (Pipeline).** Tempat Data traces logs untuk pelatihan model .                                                                                                                    |

### ---
