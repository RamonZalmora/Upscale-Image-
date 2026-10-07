# Etsy AI Image Upscaler

Aplikasi lokal untuk batch JPG / JPEG / PNG / WEBP menggunakan **Real-ESRGAN**.
Semua gambar diproses di komputer Anda. Tidak membutuhkan akun atau API berbayar.

## Windows 10 / 11

Pasang **Python 64-bit 3.11 atau 3.12** dari python.org, centang **Add Python to PATH**.
Download/clone repositori ini, ekstrak ZIP terlebih dahulu, lalu jalankan CMD:

```bat
cd /d "C:\folder Anda\Upscale-Image-"
install.bat
run.bat
```

`install.bat` membuat virtual environment, memasang dependensi, mengunduh model
resmi (~4.7 MB), memverifikasi SHA-256, dan memeriksa load model.
PyTorch Windows membutuhkan ruang disk beberapa GB dan internet saat instalasi.
`run.bat` membuka browser otomatis. Biarkan jendela CMD tetap terbuka.
Jika browser tidak terbuka, akses **http://127.0.0.1:7860** di komputer tersebut.
Menggunakan folder dengan spasi didukung. Menjalankan dari working directory lain juga didukung.

Alternatif setelah instalasi:

```bat
.venv\Scripts\activate
python app.py
```

Port lain: `run.bat --port 7861`. Tanpa membuka browser: `run.bat --no-browser`.
Tidak ada upload ke server luar. Model diunduh dari GitHub satu kali; setelah itu
aplikasi dapat dipakai offline. Model yang rusak ditolak, bukan dipakai diam-diam.

## Workflow

1. Pilih preset, scale, quality, format, DPI, dan compression.
2. **Add Images**, **Add Folder**, atau drag & drop file/folder. Add Folder mengambil
   gambar dari subfolder juga. File masuk antrean dengan settings saat ditambahkan.
3. **Start Processing**. File baru bisa ditambahkan selama proses berjalan.
4. Pause/Resume, Cancel Current/Selected, Remove Selected, Clear Queue,
   Clear Completed, dan Retry Failed tersedia. Pause/cancel berlaku pada checkpoint
   antar tile/pass, bukan memutus operasi GPU yang sedang berjalan.
5. Download setiap hasil atau **Download All ZIP**. Open File/Folder membuka file
   di komputer yang menjalankan aplikasi. Delete menghapus output setelah konfirmasi.

Untuk mengubah settings file yang sudah masuk, pilih baris **Waiting**, ubah
settings, lalu **Apply to selected waiting**. Settings file aktif tidak diubah.
Remove/Clear Queue hanya membersihkan antrean dan cache input/thumbnail; hasil final
**tetap ada**. Delete pada baris Completed menghapus hasil final di disk.
Antrean disimpan otomatis di SQLite. Setelah restart, proses yang terputus kembali
Waiting dan perlu Start Processing. Hasil Completed tetap ada.

Custom output folder menerima path absolut (mis. `D:\Etsy Assets`) atau relatif
terhadap folder project. Hasil ditulis di `<folder>/completed/`; default
`output/completed/`. Penamaan collision-safe, file asli tidak ditimpa. Custom Base
Name menggunakan nomor urut persisten seperti `moonlit-library-001.png`.

## AI dan kualitas

- Inti pipeline adalah neural inference **Real-ESRGAN realesr-general-x4v3**,
  SRVGGNetCompact resmi, bukan resize-only atau generasi ulang gambar.
- Model native **4×**. Untuk **2× / 3×**, hasil neural 4× diturunkan dengan Lanczos
  ke dimensi tepat. Lanczos bukan pengganti inference AI.
- **Fast**: 1 inference. **Balanced**: rata-rata 2 inference (horizontal flip).
  **High Quality**: rata-rata 4 inference (horizontal/vertical flip self-ensemble).
  Ini mengurangi ketidakstabilan orientasi; kecepatan tergantung ukuran/device.
- CUDA NVIDIA dipilih otomatis ketika didukung wheel PyTorch dan driver. Apple MPS
  juga didukung. Jika tidak tersedia, CPU digunakan. CPU High Quality bisa lambat
  untuk gambar besar; Fast tetap memakai AI yang sama.
- Tile AUTO berdasarkan VRAM; padding 40 pixel melebihi receptive radius model.
  CUDA OOM mengurangi tile dan akhirnya mencoba CPU. Satu gambar diproses per worker.
- Maksimum native output 100 megapixel (input sekitar 6.25 MP), demi membatasi RAM.
  Seluruh output satu gambar berada di RAM; bukan streaming tak terbatas.
- Alpha PNG/WEBP dipertahankan dan di-resample sebagai mask coverage tanpa inference
  generatif. PNG compression lossless. Ekspor transparansi ke JPG ditolak;
  pilih PNG/WEBP. RGB di-upscale AI, alpha tidak dibuat ulang oleh AI.
- JPEG memakai chroma 4:4:4. Smart compression turun bertahap dari quality 96 sampai
  floor 90/85/80. Target ukuran bersifat best effort; jika belum tercapai, aplikasi
  memberi warning dan mempertahankan kualitas. PNG tidak dikuantisasi untuk mengejar target.
- 300 DPI hanya metadata: JFIF/PNG DPI atau EXIF resolution untuk WEBP, tanpa resize
  tambahan. Sebagian viewer WEBP tidak menampilkan EXIF DPI.
- ICC profile dipertahankan jika ada. EXIF orientation diterapkan; EXIF kamera lainnya
  tidak disalin. Gambar 16-bit/animated diproses sebagai gambar RGB 8-bit/still.
- Model enhancement tidak menjamin setiap detail/font identik atau kualitas selalu
  lebih baik. Periksa hasil cetak/teks/wajah sebelum memproduksi aset untuk dijual.

Preset **ETSY PRODUCTION**: 2× High, 300 DPI, compression Balanced, target 2 MB,
Keep Original. **MAX QUALITY**: 4× High, compression off, Keep Original.

## Linux / cloud / macOS

```sh
python3 -m venv .venv
# Linux CPU-only (hindari download CUDA yang besar):
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python setup_models.py
.venv/bin/python app.py --no-browser
```

macOS: lewati perintah CPU-only dan pasang `requirements.txt` langsung.
Server hanya bind ke loopback, untuk penggunaan lokal. UI tanpa CDN/internet.
Cloud onboarding hanya memvalidasi request lokal; bukan menyediakan preview publik.

## Folder dan perawatan

```text
app.py / install.bat / run.bat
core/                 AI, export, antrean, image helpers
templates/ + static/   UI lokal dark mode
tests/                regression dan inference tests
models/               bobot model (ignored Git)
output/completed/     aset final
output/failed/        laporan kegagalan JSON
logs/app.log          rotating log
temp/uploads/         sumber antrean; disimpan untuk retry/recovery
temp/thumbnails/      preview kecil
temp/exports/         ZIP final
temp/session.sqlite3  antrean persisten
```

Preview dikirim sebagai thumbnail kecil, bukan full-resolution image ke browser.
Decoder masih perlu membaca pixel sumber saat membuat thumbnail; sumber gambar
batch tidak disimpan bersama-sama di RAM. Bersihkan entry antrean untuk menghapus
input/thumbnail yang tidak lagi diperlukan. ZIP hanya berisi output final, tidak
memasukkan model/cache/log. ZIP lama di `temp/exports/` boleh dihapus saat tidak
ada export/download aktif. Jangan hapus database/upload jika masih membutuhkan retry.

Satu instance aplikasi per folder project; instance kedua ditolak untuk melindungi antrean. Python/CUDA/model belum lengkap?
Jalankan lagi `install.bat`, lihat pesan CMD dan `logs/app.log`. Update NVIDIA driver
bila CUDA tidak terdeteksi. Jika PyTorch menampilkan DLL load error di Windows,
pasang Microsoft Visual C++ Redistributable 2015–2022 (x64). Jangan menonaktifkan TLS atau verifikasi checksum.

## Validasi

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Test suite menguji inference model asli (2×/3×/4× dan tiga mode), konsistensi tile,
PNG alpha, format JPG/WEBP, DPI, batas compression, queue/control/recovery,
collision-safe naming, upload/download, ZIP final-only, dan proteksi request lokal.
Test memakai folder sementara; output produksi tidak disentuh. Pada Windows:
`.venv\Scripts\python.exe -m unittest discover -s tests -v`.

Arsitektur/model upstream: lihat [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
