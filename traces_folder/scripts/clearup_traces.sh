#!/bin/bash

# Tanya user apakah ingin mengaktifkan opsi hapus/ganti angka 0
echo "Apakah ingin mengganti angka 0 di baris pertama menjadi 1? (y/n)"
read -r fix_zero

# Tentukan folder target (default adalah folder saat ini '.')
TARGET_DIR="./scaled_traces"

echo "Memulai pembersihan file .log di folder: $TARGET_DIR"
echo "---------------------------------------------------"

# Loop untuk setiap file .log di folder
for file in "$TARGET_DIR"/*.log; do
    # Pastikan file ada (menghindari error jika folder kosong)
    [ -e "$file" ] || continue

    echo "Memproses: $file"

    # 1. Hapus karakter Windows (CRLF) menjadi Unix (LF)
    sed -i 's/\r//g' "$file"

    # 2. Hapus spasi di awal atau akhir baris
    sed -i 's/^[[:space:]]*//;s/[[:space:]]*$//' "$file"

    # 3. Hapus baris kosong
    sed -i '/^$/d' "$file"

    # 4. Opsi ganti angka 0 di baris pertama (jika user pilih 'y')
    if [[ "$fix_zero" == "y" || "$fix_zero" == "Y" ]]; then
        sed -i '1s/^0$/1/' "$file"
        echo "   - Baris pertama 0 diganti jadi 1 (Opsional Aktif)"
    fi

done

echo "---------------------------------------------------"
echo "Selesai! Semua file log sudah bersih."
