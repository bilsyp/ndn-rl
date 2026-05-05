| Task                                   | Lokasi Pekerjaan     | Deskripsi Teknis                                                                                             |
| -------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------ |
| Task 1: Definisi Kamus Kuantisasi      | Konseptual (Global)  | Menyamakan rumus pembulatan (Bucketing) di semua sisi agar "Nama Interest" konsisten.                        |
| Task 2: Update Normalisasi Model       | AI Model (Python)    | Memperbarui fungsi normalize_observation di Python untuk mendukung 7 dimensi (termasuk CWND & RTT).          |
| Task 3: Inovasi NDN Publisher          | Server Node (Bridge) | Menambahkan logika di ndn-integrated-bridge.js untuk membuat paket Data NDN setiap kali AI memberi jawaban.  |
| Task 4: Setup Memory Store (Repo)      | Server Node (Bridge) | Mengaktifkan InMemoryDataStore di Bridge sebagai tempat penyimpanan "Keputusan AI" yang bisa diakses publik. |
| Task 5: Uji Ingesti & Freshness        | Server Node + AI     | Memastikan keputusan AI masuk ke Repo dengan FreshnessPeriod yang benar (agar memori tidak basi).            |
| Task 6: Implementasi Logika Detektif   | Client (React)       | Menambahkan fungsi getQuantizedName() di App.jsx atau NdnAbrManager.js.                                      |
| Task 7: Jalur Probing (Interest First) | Client (React)       | Mengubah alur: Klien kirim Interest ke jaringan DULU sebelum lari ke WebSocket/Bridge.                       |
| Task 8: Mekanisme Fallback             | Client (React)       | Menyiapkan logika: Jika Interest timeout (Cache Miss), otomatis pindah ke jalur WebSocket Bridge.            |
| Task 9: Safety Guard (Dua Otak)        | Client (React)       | Menambahkan filter: Jika jawaban Cache "High" tapi cwnd lokal drop, klien wajib override ke "Low".           |
| Task 10: Final Stress Test             | Semua Komponen       | Menjalankan 2-3 Klien bersamaan. Memastikan Klien ke-2 mengambil data dari Cache, bukan dari AI.             |
