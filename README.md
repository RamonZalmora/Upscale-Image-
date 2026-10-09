# Etsy AI Image Upscaler

Aplikasi lokal untuk batch JPG / JPEG / PNG / WEBP dengan pilihan **Real-ESRGAN** (detail tinggi) atau **FSRCNN** (AI ringan).
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
resmi Real-ESRGAN (~4.7 MB) dan FSRCNN (~39 KB), memverifikasi SHA-256, dan memeriksa load kedua model.
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

## Mode ringan untuk laptop dan file kecil

Pilih preset **RINGAN LAPTOP** untuk workflow hemat proses:

- **AI Ringan · detail rendah (FSRCNN)**, CPU maksimal **2 thread**, tanpa self-ensemble.
- Default **3.5×**, compression Balanced, Keep Original Format, ukuran otomatis aktif.
- Skala tersedia: **1.5×, 2×, 2.5×, 3×, 3.5×, 4×, 4.5×, 5×, 5.5×**. Ukuran piksel tidak dikurangi
  untuk mengejar target file; pecahan setengah piksel dibulatkan ke atas.
- FSRCNN adalah model neural super-resolution kecil yang berbeda dari Real-ESRGAN.
  AI native **2×**, kemudian Lanczos menyesuaikan ke skala pilihan. Hasil lebih lembut
  dan detail tambahan lebih sedikit. Skala di atas 2× tidak berarti inference AI
  native 3.5×/4.5×. Ukuran tetap mengikuti skala, tanpa menjalankan AI berulang.
- Target otomatis = **2.5× ukuran file asli, maksimal 4 MB**, minimum 0.1 MB.
  Contoh JPG 1.5 MB → target 3.75 MB. Hasil boleh lebih kecil, tidak ditambah padding.
  Matikan Ukuran Otomatis untuk memasukkan target manual. Target manual berlaku jika
  disuplai bersama auto_size melalui API; compression OFF menonaktifkan semua target.
- JPG mode ringan memakai chroma **4:2:0**, quality awal 92/86/80 untuk Light/Balanced/Maximum, pencarian biner bertarget,
  floor **85 / 72 / 65** untuk Light / Balanced / Maximum. Detail warna halus dapat
  berkurang; pilih model detail bila lebih mengutamakan ketajaman. WEBP memakai
  encoder method 2. PNG tetap lossless (compression level 4), tidak otomatis ke JPG.
- Target adalah best effort. Bila batas quality tidak cukup, output tetap disimpan
  dengan warning dan bisa lebih dari target. Gambar transparan tetap PNG/WEBP.
- Uji sintetis: JPG **1.41 MB**, 1250×1000, **3.5×** → **4375×3500**, JPG **3.76 MB**
  dengan Balanced. AI sekitar **1.3 detik** di CPU cloud pada satu pengukuran;
  hasil/kecepatan laptop dan gambar lain dapat berbeda. Ukuran file tidak berbanding
  lurus dengan skala piksel (3.5× berarti 12.25× jumlah piksel).
- Beban berkurang, tetapi laptop masih bisa hangat; mode ini tidak menjamin suhu
  tertentu. Settings model/target/skala ikut tersimpan dalam antrean dan recovery.

Untuk memperbarui instalasi lama, tutup aplikasi dan jalankan `install.bat` lagi
setelah mengganti kode (dependensi OpenCV baru diperlukan). Untuk ZIP, ekstrak ke
folder baru, jalankan `install.bat` lalu `run.bat`. Jangan menimpa/menghapus folder
output produksi Anda. Queue lama tanpa pilihan model dibaca sebagai Real-ESRGAN.

## AI dan kualitas

- Inti pipeline adalah neural inference **Real-ESRGAN realesr-general-x4v3**,
  SRVGGNetCompact resmi, bukan resize-only atau generasi ulang gambar.
- Model Real-ESRGAN native **4×**. Untuk **2× / 2.5× / 3× / 3.5×**, hasil neural 4×
  diturunkan dengan Lanczos ke dimensi tepat; **4.5×/5×/5.5×** diperbesar dari hasil AI 4×. Lanczos bukan pengganti inference AI.
- **Super Cepat (AI Turbo)**: 1 inference AI tanpa self-ensemble, tile CPU/MPS
  256 pixel dan tile CUDA sampai 768 pixel untuk mengurangi pemrosesan overlap
  berulang. CUDA memakai FP16 otomatis; CPU/MPS tetap FP32. Tile tetap memakai
  padding 40 pixel dan akan diperkecil jika memori tidak cukup. Kecepatan aktual
  tergantung perangkat dan gambar, bukan jaminan waktu tertentu. FP16 pada CUDA
  dapat sedikit mengubah hasil numerik. Transparansi, DPI, dan format tetap didukung.
  Pilih **Quality → Super Cepat**, atau terapkan ke file Waiting dengan
  **Apply to selected waiting**. Untuk throughput maksimal, matikan compression
  bila tidak diperlukan; kompresi tetap menambah waktu ekspor.
- **Fast**: 1 inference. **Balanced**: rata-rata 2 inference (horizontal flip).
  **High Quality**: rata-rata 4 inference (horizontal/vertical flip self-ensemble).
  Ini mengurangi ketidakstabilan orientasi; kecepatan tergantung ukuran/device.
- CUDA NVIDIA dipilih otomatis ketika didukung wheel PyTorch dan driver. Apple MPS
  juga didukung. Jika tidak tersedia, CPU digunakan. CPU High Quality bisa lambat
  untuk gambar besar; Fast tetap memakai AI yang sama.
- Tile AUTO berdasarkan VRAM; padding 40 pixel melebihi receptive radius model.
  CUDA OOM mengurangi tile dan akhirnya mencoba CPU. Satu gambar diproses per worker.
- Maksimum output native dan final **100 megapixel**, input maksimum **25 MP**.
  Pada model detail batas native membatasi input sekitar 6.25 MP; 4.5× membatasi
  input sekitar 4.94 MP pada kedua model.
  Seluruh output satu gambar berada di RAM; bukan streaming tak terbatas.
- Alpha PNG/WEBP dipertahankan dan di-resample sebagai mask coverage tanpa inference
  generatif. PNG compression lossless. Ekspor transparansi ke JPG ditolak;
  pilih PNG/WEBP. RGB di-upscale AI, alpha tidak dibuat ulang oleh AI.
- Pada model detail, JPEG memakai chroma 4:4:4. Smart compression turun bertahap dari quality 96 sampai
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
models/               bobot .pth dan .pb (ignored Git)
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

Test suite menguji kedua model AI asli (1.5×/2×/2.5×/3×/3.5×/4×/4.5×/5×/5.5× dan empat mode), konsistensi tile,
PNG alpha, format JPG/WEBP, DPI, batas compression, queue/control/recovery,
collision-safe naming, upload/download, ZIP final-only, dan proteksi request lokal.
Test memakai folder sementara; output produksi tidak disentuh. Pada Windows:
`.venv\Scripts\python.exe -m unittest discover -s tests -v`.

Arsitektur/model upstream: lihat [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).


## Kompresi terpisah dan panduan printable

**COMPRESS ONLY** memproses file tanpa AI dan tanpa mengubah jumlah piksel.
Pilih preset tersebut atau Proses → Compress Only. Aktifkan **File lebih kecil**
untuk JPG/WEBP pada model mana pun, pilih Maximum dan target MB bila perlu.
Profil hemat memakai JPG 4:2:0, quality awal Light/Balanced/Maximum 92/86/80
serta floor 85/72/65. Profil detail lama tetap tersedia dengan toggle hemat OFF.
PNG tetap lossless; format transparan tidak otomatis menjadi JPG. Target tidak
selalu bisa dicapai; aplikasi memberi warning. Untuk format asli dengan DPI OFF,
Compress Only mempertahankan file sumber jika encoding justru lebih besar.
DPI ON/konversi format dapat menambah metadata/ukuran, jadi pengecualian ini
memprioritaskan permintaan metadata/format pengguna.

Untuk printable, pilih preset **PRINT A4** atau **PRINT US LETTER**, tambahkan file,
pilih baris Waiting, lalu klik **Rekomendasikan skala untuk yang dipilih**.
Rekomendasi dihitung per gambar, jadi batch berbeda resolusi bisa mendapat skala berbeda.

- **A4**: 210×297 mm, target **2480×3508 px**, metadata **300 DPI**.
- **US Letter**: 8.5×11 inch, target **2550×3300 px**, metadata **300 DPI**.
- Landscape menggunakan dimensi yang ditukar. Skala terkecil yang memenuhi kedua
  sisi disarankan; misalnya Letter 1275×1650 → 2×, A4 1700×2400 → 1.5×.
- Sumber yang sudah mencapai target memakai Compress Only, tanpa memperbesar lagi.
  Jika bahkan 5.5× belum cukup, aplikasi memberi pesan, bukan menjanjikan print siap.
- Tidak ada crop/stretch/padding otomatis. Rasio kertas yang berbeda ditandai;
  cetak dengan margin agar desain tidak terpotong. Indikator PPI memakai estimasi
  konservatif untuk memenuhi kedua sisi kertas, bukan pengukuran setting printer.
- 300 PPI adalah target cetak umum, bukan jaminan penjualan, penilaian Etsy, atau
  kualitas detail sumber. Metadata 300 DPI tidak menambah detail. Periksa teks,
  garis, warna, dan hasil cetak sebelum menyatakan produk siap cetak.
- Pilihan kertas saja memberi panduan; klik rekomendasi untuk menerapkan skala
  pada Waiting. Apply to selected waiting masih tersedia untuk pengaturan manual.
