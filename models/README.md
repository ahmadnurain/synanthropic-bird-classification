# Model Klasifikasi Suara Burung Sinantropik

Folder ini berisi model hasil training yang siap digunakan untuk klasifikasi suara burung.

## File Model

| File | Deskripsi | Ukuran |
|------|-----------|--------|
| `Model_Burung_B3_GRU_v6.keras` | Model final **dengan augmentasi** (V5) — Akurasi 81,15% | ±58 MB |
| `model_baseline_final.keras` | Model baseline **tanpa augmentasi** — Akurasi 77,38% | ±58 MB |

## Detail Arsitektur

| Komponen | Spesifikasi |
|----------|-------------|
| Arsitektur | EfficientNet-B3 + GRU (256 unit) |
| Input | Citra Log-Mel Spectrogram 300×300×3 |
| Output | Probabilitas 5 kelas spesies burung sinantropik |
| Framework | TensorFlow / Keras |
| Format | `.keras` |

## Kelas Output (Label)

| Index | Nama Ilmiah | Nama Indonesia |
|-------|-------------|---------------|
| 0 | *Geopelia striata* | Perkutut Jawa |
| 1 | *Passer montanus* | Burung Gereja Erasia |
| 2 | *Pycnonotus aurigaster* | Cucak Kutilang |
| 3 | *Pycnonotus goiavier* | Merbah Cerukcuk |
| 4 | *Streptopelia chinensis* | Tekukur Biasa |

## Cara Memuat Model

```python
import tensorflow as tf

# Model dengan augmentasi (rekomendasi)
model = tf.keras.models.load_model('models/Model_Burung_B3_GRU_v6.keras')

# Model baseline (tanpa augmentasi)
model_baseline = tf.keras.models.load_model('models/model_baseline_final.keras')
```

## Perbandingan Performa

| Metrik | Baseline (Tanpa Augmentasi) | Final (Dengan Augmentasi) |
|--------|---------------------------|--------------------------|
| Akurasi Test | 77,38% | **81,15%** |
| Macro F1-Score | 0,78 | **0,81** |
| Data Train | 4.392 citra | 52.704 citra |
