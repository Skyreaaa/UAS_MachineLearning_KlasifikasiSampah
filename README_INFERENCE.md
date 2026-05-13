# Sistem Klasifikasi Sampah Hierarkis — Panduan Lengkap

Proyek ini mengklasifikasikan sampah ke dalam **2 level hierarki** secara real-time melalui kamera.

---

## Alur Pipeline

```
1_dataset_collector.py   → Kumpulkan gambar dari Bing / TrashNet
        ↓
2_preprocessor.py        → Cleaning, resize, augmentasi, split, buat label_mapping.json
        ↓
3_model_training.py      → Training model HierarchicalGarbageNet, simpan best_model.pth
        ↓
4_real_time_inference.py → Deteksi real-time via kamera dengan auto-tracking
5_inference_file.py      → Deteksi dari file image / video / folder
```

---

## Hierarki Kelas

| Level 1 | Level 2 (Subkategori) |
|---------|----------------------|
| **Organik** | daun_ranting, kayu, kertas_kardus |
| **Anorganik** | plastik, logam, kaca, tekstil, karet |
| **B3** | baterai_aki, elektronik, cat_pelarut, lampu_merkuri |

Total: **3 kelas Level 1**, **12 kelas Level 2**

---

## Script 4 — Real-Time Kamera (Auto Tracking)

```bash
cd "d:\Machine Learning UAS"
python 4_real_time_inference.py
```

### Cara Kerja Auto Tracking

1. **Warmup (40 frame)** — sistem belajar background; belum ada deteksi
2. **Foreground detection** — setiap frame dicek menggunakan MOG2 background subtraction
3. **Bounding box** — objek foreground terbesar diberi kotak dengan ukuran mengikuti objek
4. **EMA smoothing** — posisi & ukuran kotak dihaluskan agar tidak bergetar
5. **Inference** — model hanya melihat area dalam kotak, bukan seluruh frame
6. **Tampilan** — Level 1 + Level 2 + confidence % ditampilkan di dalam kotak

> Jika tidak ada objek di depan kamera → kotak hilang, muncul crosshair + hint teks.
> Jika background berubah (pindah tempat, cahaya berubah) → tekan **R** untuk reset.

### Kontrol Keyboard

| Tombol | Fungsi |
|--------|--------|
| **R** | Reset background model (ulangi warmup) |
| **C** | Ganti kamera (cycle ke kamera berikutnya) |
| **SPACE** | Capture & simpan screenshot |
| **S** | Toggle display mode: Full ↔ Minimal |
| **Q** | Quit |

### Display Mode

- **Full** (default): bar probabilitas Level 1, info kamera, frame counter, hint tombol
- **Minimal**: hanya badge kecil nama kategori di sudut kiri atas

### Tips Penggunaan

1. Pastikan kamera menghadap permukaan yang konsisten (meja, lantai)
2. Tunggu warmup selesai sebelum menaruh objek
3. Pencahayaan cukup → confidence lebih tinggi
4. Jarak ideal: 20–50 cm dari objek
5. Jika box sering muncul di noise background → tekan R sambil frame kosong

---

## Script 5 — Deteksi File / Gambar / Video

```bash
cd "d:\Machine Learning UAS"

# Single image
python 5_inference_file.py --image path/ke/gambar.jpg

# Video file
python 5_inference_file.py --video path/ke/video.mp4

# Folder berisi gambar
python 5_inference_file.py --folder path/ke/folder/
```

---

## Arsitektur Model — HierarchicalGarbageNet

```
Input (224×224×3)
    │
EfficientNet-B0 (backbone, ImageNet pretrained)
    │  fitur: 1280-dim
shared_fc: Linear(1280→512) + BatchNorm + ReLU + Dropout(0.4)
    │  fitur: 512-dim
    ├──► head_lvl1: Linear(512→128) → ReLU → Dropout → Linear(128→3)   → Level 1
    └──► head_lvl2: Linear(512→256) → ReLU → Dropout → Linear(256→12)  → Level 2
```

**Loss Function:**
```
total_loss = 0.4 × CrossEntropy(Level 1)
           + 0.6 × CrossEntropy(Level 2)
           + 0.1 × Consistency Loss
```

Consistency Loss memastikan prediksi Level 2 konsisten dengan Level 1, dengan cara memproyeksikan probabilitas Level 2 ke ruang Level 1 lalu menghitung CrossEntropy terhadap label Level 1.

**Training Strategy:**
- Epoch 1–9 : Backbone frozen, hanya head yang dilatih (LR = 1e-4)
- Epoch 10+ : Backbone di-unfreeze, fine-tuning seluruh jaringan (LR = 5e-5)
- Optimizer : AdamW + weight decay 1e-4
- Scheduler : CosineAnnealingLR (LR turun ke 1e-6)
- Early stopping : patience = 6 epoch

**Parameter:**
- Total : ~4.86 juta
- Trainable (awal) : ~857K (17.6%) — backbone frozen

---

## Struktur File

```
d:\Machine Learning UAS\
├── 1_dataset_collector.py      Scraping gambar dari Bing / TrashNet
├── 2_preprocessor.py           Cleaning, augmentasi, split, label mapping
├── 3_model_training.py         Training model hierarkis
├── 4_real_time_inference.py    Deteksi real-time via kamera (auto tracking)
├── 5_inference_file.py         Deteksi dari file image / video / folder
├── diagnose_camera.py          Tool diagnostik masalah kamera
├── model\
│   ├── best_model.pth          Model terbaik (berdasarkan val_acc_lvl1)
│   ├── resume_checkpoint.pth   Checkpoint untuk resume training (per epoch)
│   ├── confusion_matrix.png    Visualisasi confusion matrix test set
│   └── training_history.png    Grafik loss & accuracy selama training
└── dataset\
    ├── label_mapping.json      Mapping kelas (dibuat oleh 2_preprocessor.py)
    ├── distribusi_dataset.png  Grafik distribusi dataset
    ├── raw\                    Dataset mentah per subkelas
    └── processed\              Dataset setelah preprocessing & split
```

---

## Requirements

```bash
pip install torch torchvision timm opencv-python Pillow scikit-learn matplotlib seaborn tqdm albumentations icrawler
```

| Package | Kegunaan |
|---------|----------|
| torch, torchvision | Deep learning framework |
| timm | EfficientNet-B0 backbone |
| opencv-python | Kamera & image processing (gunakan `opencv-python`, bukan `headless`) |
| Pillow | Baca/tulis gambar |
| scikit-learn | Classification report, confusion matrix |
| matplotlib, seaborn | Visualisasi training & evaluasi |
| albumentations | Augmentasi gambar di preprocessor |
| icrawler | Scraping gambar Bing di dataset collector |

---

## Hardware

| Komponen | Minimum | Rekomendasi |
|----------|---------|-------------|
| CPU | Any | Modern multi-core |
| GPU | Tidak wajib | CUDA GPU (untuk inference lebih cepat) |
| RAM | 4 GB | 8 GB+ |
| Kamera | USB webcam | 1080p webcam |

---

## Troubleshooting

### Kamera tidak terdeteksi
```bash
python diagnose_camera.py
```
- Pastikan kamera terpasang & tidak digunakan aplikasi lain (Zoom, Teams, dll)
- Tekan `C` di dalam aplikasi untuk mencoba kamera lain

### `cv2.imshow not implemented`
Tanda `opencv-python-headless` ter-install, bukan versi full:
```bash
pip uninstall opencv-python-headless opencv-python -y
pip install opencv-python
```

### Model tidak ditemukan
Pastikan `model/best_model.pth` ada. Jika belum, jalankan training:
```bash
python 3_model_training.py
```

### Training terhenti lalu mau lanjut
Script otomatis resume dari `model/resume_checkpoint.pth` jika file tersebut ada.
Untuk mulai dari awal: hapus `resume_checkpoint.pth` dan `best_model.pth` dulu.

### Confidence rendah / deteksi tidak akurat
- Pastikan pencahayaan cukup terang
- Objek terlihat jelas dan tidak terpotong
- Coba dari sudut berbeda
- Tekan `R` jika box muncul di lokasi yang salah
