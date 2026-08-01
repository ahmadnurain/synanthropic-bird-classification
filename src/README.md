# Skrip Program Penelitian

Skrip penelitian dikelompokkan ke dalam 3 folder berdasarkan fungsinya agar mudah dipahami dan dijalankan.

## Urutan Penggunaan

### 📁 `preprocessing/` — Pipeline Preprocessing (Dipakai oleh Kedua Skenario)

Skrip-skrip ini dijalankan secara berurutan untuk mempersiapkan data audio mentah hingga menjadi citra spectrogram. **Kedua skenario** (dengan dan tanpa augmentasi) membutuhkan skrip ini.

| Urutan | Skrip | Fungsi |
|--------|-------|--------|
| 1 | `download_xc.py` | Mengunduh metadata dan audio dari API Xeno-canto |
| 2 | `calculate_dration_balanced.py` | Menghitung total durasi per spesies |
| 3 | `balance_by_duration.py` | Menyeimbangkan dataset berbasis durasi (5.250 detik/spesies) |
| 4 | `quality_control.py` | Quality control audio (RMS Energy, SNR) |
| 5 | `audio_pipeline.py` | File-level split → Resample 32kHz → Mono → Normalisasi → Segmentasi 5 detik → RMS Filtering |
| 6 | `generate_spectrograms.py` | Konversi segmen WAV → Log-Mel Spectrogram PNG |

### 📁 `dengan_augmentasi/` — Skrip Khusus Skenario Dengan Augmentasi

Skrip tambahan yang **hanya dijalankan untuk skenario dengan augmentasi**. Dijalankan setelah preprocessing selesai.

| Urutan | Skrip | Fungsi |
|--------|-------|--------|
| 7a | `augment_environmental_noise.py` | Menambahkan noise angin & hujan sintetis pada audio Train (SNR 10–20 dB) |
| 7b | `specaugment_local.py` | SpecAugment pada gambar spectrogram Train (Freq Masking + Time Masking + Gabungan) |

Hasil: Data Train naik dari 4.392 → **52.704 citra**.

### 📁 `tanpa_augmentasi/` — Skrip Khusus Skenario Baseline (Tanpa Augmentasi)

Skrip alternatif yang **hanya dijalankan untuk skenario baseline** (tanpa augmentasi apapun).

| Urutan | Skrip | Fungsi |
|--------|-------|--------|
| 7a | `prepare_baseline_dataset.py` | Mempersiapkan struktur dataset baseline |
| 7b | `generate_baseline_spectrograms.py` | Generate spectrogram tanpa augmentasi |

Hasil: Data Train tetap **4.392 citra** (tidak ada penambahan).

## Diagram Alur

```
                    preprocessing/ (1-6)
                           |
              ┌────────────┴────────────┐
              ▼                         ▼
    dengan_augmentasi/ (7)     tanpa_augmentasi/ (7)
              |                         |
              ▼                         ▼
   notebooks/Training_*        notebooks/Training_*
   DENGAN AUGMENTASI.ipynb     Baseline.ipynb
              |                         |
              ▼                         ▼
     Akurasi: 81,15%           Akurasi: 77,38%
```
