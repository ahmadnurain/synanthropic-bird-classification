# AvesIdent — Klasifikasi Suara Burung Sinantropik

Repositori penelitian **Tugas Akhir / Skripsi** — Klasifikasi Suara Burung Sinantropik Menggunakan Arsitektur **EfficientNet-B3 + GRU** pada Citra **Log-Mel Spectrogram**.

## Ringkasan Penelitian

Penelitian ini mengembangkan sistem klasifikasi otomatis untuk mengidentifikasi **5 spesies burung sinantropik** berdasarkan suara (vokalisasi) mereka. Data audio diperoleh dari repositori publik [Xeno-canto](https://xeno-canto.org/), diproses menjadi citra Log-Mel Spectrogram, lalu diklasifikasikan menggunakan model _Deep Learning_ gabungan **EfficientNet-B3** (ekstraksi fitur visual) dan **GRU** (pemodelan pola temporal).

### Spesies Target

| No | Nama Ilmiah | Nama Indonesia |
|----|-------------|---------------|
| 1 | *Geopelia striata* | Perkutut Jawa |
| 2 | *Passer montanus* | Burung Gereja Erasia |
| 3 | *Pycnonotus aurigaster* | Cucak Kutilang |
| 4 | *Pycnonotus goiavier* | Merbah Cerukcuk |
| 5 | *Streptopelia chinensis* | Tekukur Biasa |

### Hasil Evaluasi

| Metrik | Model Baseline (Tanpa Augmentasi) | Model Final (Dengan Augmentasi) |
|--------|-----------------------------------|----------------------------------|
| **Akurasi Test** | 77,38% | **81,15%** |
| **Macro F1-Score** | 0,78 | **0,81** |
| **Data Train** | 4.392 citra | 52.704 citra |

## Dataset

Dataset penelitian tersedia di **Google Drive** (berisi data spectrogram yang sudah melewati seluruh proses augmentasi dan data audio yang sudah di-balance durasinya):

> 📁 **[Download Dataset Penelitian](https://drive.google.com/drive/folders/1k_585YMj5N318KMyVi1Ic4FKn-B3VitE?usp=sharing)**

Isi dataset:

| Folder | Isi | Keterangan |
|--------|-----|------------|
| `dataset_spectrograms/` | 53.774 citra PNG Log-Mel Spectrogram | Dataset final untuk training (train/val/test), sudah termasuk SpecAugment |
| `dataset_duration_balanced/` | 628 file audio WAV | Audio mentah yang sudah di-balance durasinya (5.250 detik/spesies) |

> **Catatan:** Data audio mentah dari Xeno-canto tidak disertakan karena ukurannya sangat besar (~9 GB). Data tersebut dapat diunduh ulang menggunakan skrip `src/preprocessing/download_xc.py`.

## Struktur Repositori

```
avesident/
├── README.md                          ← Dokumentasi ini
├── requirements.txt                   ← Dependensi Python
├── .gitignore
│
├── src/                               ← Semua skrip penelitian
│   ├── README.md                      ← Panduan urutan penggunaan skrip
│   │
│   ├── preprocessing/                 ← Pipeline preprocessing (kedua skenario)
│   │   ├── download_xc.py             ← Unduh data dari Xeno-canto
│   │   ├── calculate_dration_balanced.py ← Hitung durasi per spesies
│   │   ├── balance_by_duration.py     ← Penyeimbangan dataset berbasis durasi
│   │   ├── quality_control.py         ← Quality control audio (RMS, SNR)
│   │   ├── audio_pipeline.py          ← Split, resample, segmentasi, RMS filtering
│   │   └── generate_spectrograms.py   ← Konversi audio → Log-Mel Spectrogram
│   │
│   ├── dengan_augmentasi/             ← Skrip khusus skenario DENGAN augmentasi
│   │   ├── augment_environmental_noise.py  ← Noise angin & hujan (SNR 10-20 dB)
│   │   └── specaugment_local.py       ← SpecAugment (freq/time masking)
│   │
│   └── tanpa_augmentasi/              ← Skrip khusus skenario TANPA augmentasi
│       ├── generate_baseline_spectrograms.py  ← Spectrogram baseline
│       └── prepare_baseline_dataset.py        ← Persiapan dataset baseline
│
├── notebooks/                         ← Notebook Google Colab
│   ├── Training_EfficientNetB3_DENGAN AUGMENTASI.ipynb
│   └── Training_Baseline_TanpaAugmentasi berhasil di pakai.ipynb
│
├── metadata/                          ← Data metadata penelitian
│   ├── metadata_raw_xeno_canto.csv    ← Metadata mentah 3.635 rekaman Xeno-canto
│   ├── metadata_final_dataset.csv     ← Metadata akhir + status penelitian (digunakan/tidak)
│   ├── audio_qc_report.csv            ← Laporan QC audio per file
│   └── ground_truth_frekuensi.csv     ← Ground truth frekuensi spesies
│
├── results/                           ← Hasil evaluasi model
│   ├── dengan_augmentasi/             ← Model V5 (final)
│   │   ├── classification_report_v6.txt
│   │   ├── confusion_matrix_v6.png
│   │   └── training_history_v6.png
│   └── tanpa_augmentasi/              ← Model Baseline
│       ├── classification_report_baseline.txt
│       ├── confusion_matrix_baseline.png
│       └── training_curves_baseline.png
│
├── visualisasi/                       ← Gambar visualisasi untuk skripsi
│   ├── Gambar_3_3_Waveform_Mentah.png
│   ├── Gambar_3_18_Perbandingan_Spectrogram.png
│   ├── Gambar_4_3_Contoh_LogMel_Per_Spesies.png
│   └── ... (15 gambar)
│
├── models/                            ← Model hasil training (.keras)
│   ├── Model_Burung_B3_GRU_v6.keras   (58 MB, dengan augmentasi)
│   ├── model_baseline_final.keras     (58 MB, tanpa augmentasi)
│   └── README.md
│
└── docs/                              ← Dokumentasi metodologi
    └── Laporan_Preprocessing_Audio2.md
```

## Pipeline Penelitian

```
                    preprocessing/ (langkah 1-6)
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    dengan_augmentasi/ (7)     tanpa_augmentasi/ (7)
              │                         │
              ▼                         ▼
   notebooks/Training_*        notebooks/Training_*
   DENGAN AUGMENTASI.ipynb     Baseline.ipynb
              │                         │
              ▼                         ▼
     Akurasi: 81,15%           Akurasi: 77,38%
```

### Detail Langkah

```
1. Pengumpulan Data (Xeno-canto)          → preprocessing/download_xc.py
         ↓
2. Penyeimbangan Durasi                   → preprocessing/balance_by_duration.py
         ↓
3. Preprocessing Audio                    → preprocessing/audio_pipeline.py
   (Mono, 32kHz, Normalisasi, Segmentasi 5 detik, RMS Filtering)
         ↓
4. Konversi ke Log-Mel Spectrogram        → preprocessing/generate_spectrograms.py
   (128 mel bands, 500-12500 Hz, RGB PNG)
         ↓
   ┌─── Skenario DENGAN Augmentasi ───┐   ┌─── Skenario TANPA Augmentasi ───┐
   │ 5. Noise Angin & Hujan           │   │ 5. Langsung ke training         │
   │    → dengan_augmentasi/           │   │    → tanpa_augmentasi/           │
   │      augment_environmental_noise  │   │      prepare_baseline_dataset    │
   │ 6. SpecAugment                    │   │      generate_baseline_spectro   │
   │    → specaugment_local.py         │   │                                  │
   │ Data Train: 52.704 citra          │   │ Data Train: 4.392 citra          │
   └───────────────────────────────────┘   └──────────────────────────────────┘
         ↓                                         ↓
7. Training EfficientNet-B3 + GRU         → notebooks/*.ipynb
   (2 fase: Feature Extraction → Fine-Tuning)
```

## Ringkasan Dataset

| Komponen | Nilai |
|----------|-------|
| Sumber data | Repositori publik Xeno-canto |
| Jumlah spesies | 5 spesies sinantropik |
| Total data mentah | 3.409 file (±65 jam 35 menit) |
| Target durasi balancing | 5.250 detik per spesies |
| Segmen asli setelah preprocessing | 5.462 segmen (5 detik/segmen) |
| Data Train (asli) | 4.392 segmen |
| Data Train (setelah augmentasi) | 52.704 citra |
| Data Validasi | 566 citra |
| Data Uji | 504 citra |

## Teknologi

- **Python 3.x** — Bahasa pemrograman utama
- **librosa** — Pemrosesan audio dan ekstraksi fitur
- **TensorFlow / Keras** — Framework Deep Learning
- **EfficientNet-B3** — Backbone CNN (pre-trained ImageNet)
- **GRU** — Recurrent layer untuk pola temporal
- **Google Colab** — Platform training (GPU NVIDIA T4)

## Model

Model klasifikasi disertakan langsung di folder `models/`:

| File | Deskripsi | Akurasi |
|------|-----------|---------| 
| `Model_Burung_B3_GRU_v6.keras` | Model final (dengan augmentasi) | **81,15%** |
| `model_baseline_final.keras` | Model baseline (tanpa augmentasi) | 77,38% |

Lihat `models/README.md` untuk detail arsitektur dan cara penggunaan.

## Referensi Utama

- Baowaly, M. K. et al. (2024). *Bird species classification from audio using EfficientNet and GRU.*
- Kumar, V. et al. (2024). *Improving learning-based birdsong classification by utilizing combined audio augmentation strategies.* Ecological Informatics.
- Park, D. S. et al. (2019). *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.*

## Lisensi

Repositori ini dibuat untuk keperluan akademis (Tugas Akhir/Skripsi). Data audio bersumber dari [Xeno-canto](https://xeno-canto.org/) yang didistribusikan di bawah lisensi Creative Commons.
