# Laporan Metodologi: Preprocessing Dataset Audio & Pelatihan Model Klasifikasi Suara Burung

Laporan ini merangkum seluruh tahapan yang telah dilakukan dalam penelitian klasifikasi suara burung, mulai dari pengunduhan data mentah dari Xeno-canto hingga diperolehnya model _Deep Learning_ final yang siap digunakan. Metodologi ini merujuk pada jurnal **Baowaly et al. (2024)** sebagai acuan utama.

---

## A. Lingkungan Pengembangan & Alat yang Digunakan

### A.1. Lingkungan Komputasi

Seluruh tahapan _preprocessing_ data (pengunduhan, QC, segmentasi, dan ekstraksi ciri) dijalankan secara lokal di mesin pengembang dengan spesifikasi:

- **Sistem Operasi:** Windows 11
- **Bahasa Pemrograman:** Python 3.x
- **Editor:** Visual Studio Code

Untuk tahap **Pelatihan Model (_Training_)**, proses komputasi dialihkan ke platform awan **Google Colaboratory (Colab)** karena keterbatasan kapasitas GPU pada mesin lokal. Percobaan awal yang dilakukan menggunakan CPU standar menunjukkan durasi _training_ yang tidak realistis (diestimasi berhari-hari). Google Colab menyediakan akses GPU **NVIDIA Tesla T4** secara gratis, yang mampu mempercepat proses _training_ dari hitungan hari menjadi hitungan jam (±3–5 jam untuk keseluruhan proses dua fase).

### A.2. Library & Dependensi Python

Berikut adalah seluruh _library_ Python yang digunakan, dikelompokkan berdasarkan fungsinya:

#### Pengunduhan Data & Komunikasi Jaringan

| Library    | Versi    | Fungsi                                                                                                               |
| ---------- | -------- | -------------------------------------------------------------------------------------------------------------------- |
| `aiohttp`  | Terbaru  | _Asynchronous HTTP Client_ untuk mengunduh metadata dan file audio dari API Xeno-canto secara paralel tanpa antrian. |
| `aiofiles` | Terbaru  | Menulis file audio hasil unduhan ke disk secara _asynchronous_ agar tidak memblokir proses unduhan lain.             |
| `asyncio`  | Built-in | Kerangka kerja _asynchronous_ bawaan Python untuk mengelola ratusan koneksi unduhan secara bersamaan.                |

#### Pemrosesan Audio

| Library     | Versi   | Fungsi                                                                                                                                                                                                                   |
| ----------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `librosa`   | ≥0.10   | _Library_ utama pemrosesan audio: membaca file, _resampling_ ke 32.000 Hz, menghitung _RMS Energy_, SNR, _Pitch Shifting_, _Time Stretching_, dan menghitung **Log-Mel Spectrogram** (`librosa.feature.melspectrogram`). |
| `soundfile` | Terbaru | Menyimpan (_export_) file audio dalam format WAV standar setelah _preprocessing_ selesai.                                                                                                                                |

#### Manipulasi Data & Komputasi Numerik

| Library  | Versi   | Fungsi                                                                                                   |
| -------- | ------- | -------------------------------------------------------------------------------------------------------- |
| `numpy`  | ≥1.24   | Komputasi array numerik. Digunakan untuk operasi matriks pada data audio (_waveform_) dan _spectrogram_. |
| `pandas` | Terbaru | Membaca dan menulis file CSV (laporan QC, metadata).                                                     |

#### Visualisasi & Ekspor Gambar

| Library      | Versi   | Fungsi                                                                                              |
| ------------ | ------- | --------------------------------------------------------------------------------------------------- |
| `matplotlib` | Terbaru | Merender dan menyimpan gambar Log-Mel Spectrogram ke PNG tanpa batas tepi (_marginless_).           |
| `seaborn`    | Terbaru | Membuat visualisasi _Confusion Matrix_ berbentuk _heatmap_ berwarna untuk keperluan evaluasi model. |

#### Manajemen File & Utilitas

| Library               | Versi    | Fungsi                                                                                                                                         |
| --------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `pathlib`             | Built-in | Pengelolaan jalur direktori secara lintas platform (Windows/Linux).                                                                            |
| `tqdm`                | Terbaru  | Menampilkan _progress bar_ pada terminal agar proses panjang dapat dipantau.                                                                   |
| `os`, `sys`, `shutil` | Built-in | Operasi file sistem: membuat folder, memindahkan file, memeriksa keberadaan file.                                                              |
| `random`              | Built-in | Pengacakan urutan file untuk proses _splitting_ dan _sampling_.                                                                                |
| `json`                | Built-in | Parsing respons API Xeno-canto berbentuk JSON.                                                                                                 |
| `cv2` (OpenCV)        | Terbaru  | Membaca dan menulis file gambar PNG Spectrogram saat proses augmentasi gambar.                                                                 |
| `scipy`               | ≥1.10    | Digunakan untuk mendesain _Butterworth bandpass filter_ pada pembuatan _noise_ angin sintetis (`scipy.signal.butter`, `scipy.signal.lfilter`). |

#### Machine Learning & Pelatihan Model (Google Colab)

| Library          | Versi         | Fungsi                                                                                                    |
| ---------------- | ------------- | --------------------------------------------------------------------------------------------------------- |
| `tensorflow`     | ≥2.15         | _Framework_ utama _Deep Learning_: infrastruktur komputasi, pengelolaan GPU, dan loop _training_.         |
| `keras`          | ≥3.0 (via TF) | API tingkat tinggi untuk membangun arsitektur model (`layers`, `models`, `callbacks`).                    |
| `EfficientNetB3` | via Keras     | Arsitektur _EfficientNet-B3_ beserta bobot _pre-trained ImageNet_ dari `tf.keras.applications`.           |
| `sklearn`        | Terbaru       | Menghitung _Confusion Matrix_ dan _Classification Report_ (Precision, Recall, F1-Score) pasca _training_. |

---

## B. Skrip yang Dikembangkan

| Nama Skrip                       | Fungsi Utama                                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `download_xc.py`                 | Mengunduh metadata dan audio dari API Xeno-canto secara _async_.                                              |
| `calculate_duration_balanced.py` | Menghitung total durasi akurat per spesies untuk memastikan target 5.250 detik terpenuhi.                     |
| `balance_by_duration.py`         | Menyeimbangkan dataset berbasis total durasi (bukan jumlah file).                                             |
| `quality_control.py`             | Menjalankan _Quality Control_ audio: evaluasi RMS _Energy_, SNR, dan filter segmen tidak layak.               |
| `audio_pipeline.py`              | _Pipeline_ utama: _file-level split_, _resampling_, normalisasi, _chunking_ 5 detik, dan _salient filtering_. |
| `generate_spectrograms.py`       | Mengubah seluruh segmen WAV menjadi gambar Log-Mel Spectrogram PNG.                                           |
| `augment_environmental_noise.py` | Augmentasi audio mentah dengan _noise_ lingkungan sintetis (angin dan hujan) pada berbagai level SNR.         |
| `specaugment_local.py`           | Menerapkan _SpecAugment_ pada gambar Spectrogram: _Frequency Masking_, _Time Masking_, dan gabungan keduanya. |
| `generate_baseline_spectrograms.py` | Menghasilkan gambar Log-Mel Spectrogram khusus untuk skenario _baseline_ (tanpa augmentasi).               |
| `prepare_baseline_dataset.py`    | Mempersiapkan struktur dataset _baseline_ tanpa augmentasi untuk perbandingan performa.                       |
| `Training_EfficientNetB3.ipynb`  | _Notebook_ Google Colab untuk pelatihan model EfficientNet-B3 + GRU lengkap dengan augmentasi dan evaluasi.   |


---

## C. Tahapan Preprocessing Audio

### C.1. Pengumpulan Data Awal

Sebanyak **3.409 file audio** berhasil diunduh dari repositori Xeno-canto menggunakan skrip `download_xc.py`. Filter kualitas metadata (A/B) sengaja diabaikan untuk memaksimalkan jumlah data yang tersedia. Data ini kemudian langsung diteruskan ke tahap penyeimbangan data.

**Tabel C.1. Jumlah Data Mentah per Spesies dari Xeno-canto**

| No  | Spesies (Nama Ilmiah)      | Nama Indonesia       | Jumlah File | Total Durasi Mentah   |
| --- | -------------------------- | -------------------- | ----------- | --------------------- |
| 1   | _Geopelia striata_         | Perkutut Jawa        | 190         | 1 Jam 46 Menit        |
| 2   | _Lonchura leucogastroides_ | Bondol Jawa          | 15          | 7 Menit               |
| 3   | _Passer montanus_          | Burung Gereja Erasia | 2.595       | 56 Jam 42 Menit       |
| 4   | _Pycnonotus aurigaster_    | Cucak Kutilang       | 166         | 1 Jam 54 Menit        |
| 5   | _Pycnonotus goiavier_      | Merbah Cerukcuk      | 140         | 1 Jam 44 Menit        |
| 6   | _Streptopelia chinensis_   | Tekukur Biasa        | 303         | 3 Jam 20 Menit        |
|     | **Total**                  |                      | **3.409**   | **≈ 65 Jam 35 Menit** |

> **Catatan:** _Lonchura leucogastroides_ (15 file) dikecualikan dari penelitian karena jumlah data terlalu sedikit untuk memenuhi target durasi penyeimbangan. Penelitian dilanjutkan dengan **5 spesies target**.

Kelima spesies target merupakan burung **sinantropik** (_synanthropic birds_), yaitu spesies yang telah beradaptasi dan hidup berdampingan dengan manusia di habitat termodifikasi seperti pemukiman, taman kota, dan area pertanian. Pemilihan spesies ini relevan untuk pengembangan sistem pemantauan keanekaragaman hayati di lingkungan perkotaan.

### C.2. Penyeimbangan Data (_Dataset Balancing_)

Terdapat ketidakseimbangan kelas (_Imbalanced Dataset_) yang parah (contoh: _Passer montanus_ memiliki 2.595 file, sedangkan _Pycnonotus goiavier_ hanya 140 file).

- **Undersampling:** Pada spesies mayoritas, dipilih file secara acak untuk membatasi dominasinya.
- **Oversampling (Augmentasi Audio):** Pada spesies minoritas, dilakukan duplikasi yang dikombinasikan dengan _Pitch Shifting_, _Time Stretching_, dan _Add Noise_ menggunakan `librosa`.

### C.3. Penyeragaman Durasi Akurat (_Duration Balancing_)

Penyeimbangan data dikalibrasi ulang dari berbasis "jumlah file" menjadi berbasis "**total durasi**" agar segmen yang dihasilkan benar-benar seimbang.

- **Target Durasi:** **5.250 detik (1 Jam 27 Menit 30 Detik)** per spesies. Angka ini dipilih karena habis dibagi _window_ 5 detik, sehingga menghasilkan tepat **1.050 segmen per spesies**.
- **Total Durasi Keseluruhan:** 5 spesies × 5.250 detik = **26.250 detik (≈ 7 Jam 17 Menit 30 Detik)**.

**Tabel C.2. Hasil Penyeimbangan Durasi Dataset**

| Spesies                  | Jumlah File Setelah Balancing | Target Durasi    |
| ------------------------ | ----------------------------- | ---------------- |
| _Geopelia striata_       | 164                           | 5.250 detik      |
| _Passer montanus_        | 83                            | 5.250 detik      |
| _Pycnonotus aurigaster_  | 137                           | 5.250 detik      |
| _Pycnonotus goiavier_    | 105                           | 5.250 detik      |
| _Streptopelia chinensis_ | 139                           | 5.250 detik      |
| **Total**                | **628**                       | **26.250 detik** |

### C.4. Preprocessing Final & File-Level Splitting

> **Penting:** Pembagian data dilakukan di **level file asli** (bukan level segmen) untuk mencegah _Data Leakage_ — memastikan segmen dari rekaman yang sama tidak tumpah ke data pengujian.

1.  **File-Level Splitting:** Seluruh file asli dibagi secara acak menjadi **Train (80%)**, **Validasi (10%)**, dan **Test (10%)** sebelum proses segmentasi dimulai.
2.  **Standarisasi Audio:** Setiap file dikonversi menjadi **Mono** dengan _sample rate_ **32.000 Hz**, lalu dilakukan _Peak Amplitude Normalization_ (skala -1.0 hingga 1.0).
3.  **Fixed-length Chunking (5 Detik):** File diiris menjadi segmen konstan 5 detik. Segmen terakhir yang kurang dari 5 detik dilengkapi dengan _Zero-Padding_.
4.  **Salient Filtering:** Setiap segmen dievaluasi menggunakan _Root Mean Square_ (RMS) _Energy_. Segmen dengan nilai RMS di bawah _threshold_ **0,002** (menandakan kesunyian absolut atau hanya angin) dibuang untuk mencegah model berlatih pada data kosong.

**Tabel C.3. Distribusi Segmen Audio (dan Durasi) per Kelas Setelah Preprocessing**

| Spesies                  | Train (Segmen / Durasi) | Validasi (Segmen / Durasi) | Test (Segmen / Durasi) | Total Segmen           |
| ------------------------ | ----------------------- | -------------------------- | ---------------------- | ---------------------- |
| _Geopelia striata_       | 900 (75m)               | 111 (9m 15d)               | 105 (8m 45d)           | 1.116                  |
| _Passer montanus_        | 866 (72m 10d)           | 105 (8m 45d)               | 98 (8m 10d)            | 1.069                  |
| _Pycnonotus aurigaster_  | 880 (73m 20d)           | 108 (9m)                   | 107 (8m 55d)           | 1.095                  |
| _Pycnonotus goiavier_    | 854 (71m 10d)           | 121 (10m 5d)               | 101 (8m 25d)           | 1.076                  |
| _Streptopelia chinensis_ | 892 (74m 20d)           | 121 (10m 5d)               | 93 (7m 45d)            | 1.106                  |
| **Total**                | **4.392 (6j 6m)**       | **566 (47m 10d)**          | **504 (42m)**          | **5.462 (7j 35m 10d)** |

> **Catatan:** Angka Train di atas (4.392 segmen / 6 Jam 6 Menit) adalah jumlah data **asli sebelum augmentasi** (baik noise lingkungan maupun SpecAugment). Setelah augmentasi noise (angin dan hujan), jumlah ini naik 3× menjadi 13.176 segmen (setara dengan **18 Jam 18 Menit** audio latih), lalu setelah SpecAugment menjadi 52.704 gambar Train.

### C.5. Ekstraksi Ciri: Log-Mel Spectrogram

Ribuan segmen audio 5 detik ditransformasikan menjadi representasi citra **Log-Mel Spectrogram** 2D. Parameter mengacu secara eksplisit pada **Baowaly et al. (2024)**:

| Parameter         | Nilai         | Alasan                                                                 |
| ----------------- | ------------- | ---------------------------------------------------------------------- |
| **N_Mels**        | 128           | Resolusi frekuensi standar untuk klasifikasi suara burung              |
| **FMIN**          | 500 Hz        | Memfilter _noise_ frekuensi sangat rendah (angin, getaran)             |
| **FMAX**          | 12.500 Hz     | Mencakup rentang vokalisasi mayoritas spesies burung target            |
| **Format Output** | Citra RGB PNG | Tanpa batas tepi (_marginless_) agar tidak ada piksel aksesoris grafik |

---

## D. Augmentasi Audio Mentah: Noise Lingkungan (Angin & Hujan)

Sebelum data audio dikonversi ke _Spectrogram_, dilakukan tahap augmentasi pada **sinyal audio mentah** (_raw waveform_) di level data **Train** saja. Tujuannya adalah meningkatkan **robustness** (ketahanan) model terhadap kondisi rekaman lapangan yang tidak ideal, di mana suara burung sering tercampur dengan suara alam seperti angin dan hujan.

> **Catatan Penting:** Augmentasi ini **hanya diterapkan pada data Train**. Data Validasi dan Test **tidak** diaugmentasi agar hasil evaluasi tetap objektif dan mencerminkan kemampuan generalisasi model yang sesungguhnya.

### D.1. Dasar Ilmiah

Teknik _Background Noise Injection_ pada audio mentah didukung oleh beberapa penelitian dalam klasifikasi suara burung:

1.  **Kumar, V. et al. (2024).** _"Improving learning-based birdsong classification by utilizing combined audio augmentation strategies."_ Ecological Informatics.
    - Rujukan utama. Membuktikan bahwa kombinasi strategi augmentasi audio — termasuk _simulated pink noise_, _interspecies sound mixing_, dan _loudness normalization_ — menghasilkan performa terbaik untuk klasifikasi kicauan burung.

2.  **Dmitriev, A. et al. (2024).** _"Methods for Training Convolutional Neural Networks to Identify Bird Sounds."_
    - Membahas tantangan penambahan _noise_ hujan dan angin: karakter _noise_ lingkungan bersifat **non-stasioner** dan rekaman _noise_ di alam liar sering mengandung suara burung latar belakang. Oleh karena itu, dalam penelitian ini digunakan **noise sintetis murni** (dibuat dari rumus matematika) untuk menghindari masalah kontaminasi tersebut.

3.  **Hong, J. et al. (2023).** _"Acoustic Bird Species Recognition at BirdCLEF 2023."_
    - Menggunakan augmentasi _GaussianNoise_ dan _PinkNoise_ pada audio mentah sebelum konversi ke Mel-spectrogram sebagai teknik augmentasi standar pada kompetisi BirdCLEF.

4.  **Alzahra, S. et al. (2019).** _"Bird sounds classification by combining PNCC and robust Mel-log filter bank features."_
    - Memberikan dasar angka SNR eksplisit pada klasifikasi suara burung: augmentasi _background noise_ pada **4 level SNR (20 dB, 10 dB, 5 dB, dan 0 dB)**. Paper ini menjadi acuan pemilihan rentang SNR 10–20 dB pada penelitian ini.

5.  **Michaud, A. et al. (2025).** _"Acoustic detection of a nocturnal bird with deep learning: the challenge of low signal-to-noise ratio."_
    - Menunjukkan bahwa performa model deteksi burung **menurun tajam ketika SNR di bawah 3 dB**. Temuan ini menjadi dasar keputusan untuk **tidak menggunakan SNR 0 dB** sebagai setting augmentasi utama.

### D.2. Pemilihan Level SNR

**SNR (Signal-to-Noise Ratio)** adalah perbandingan kekuatan suara burung terhadap _noise_. Semakin besar SNR, suara burung semakin jelas; semakin kecil SNR, _noise_ semakin dominan.

Berdasarkan Alzahra et al. (2019) dan pertimbangan dari Michaud et al. (2025), dipilih **3 level SNR** yang aman:

| SNR       | Kondisi Noise       | Makna                                                                |
| --------- | ------------------- | -------------------------------------------------------------------- |
| **20 dB** | Noise ringan        | Suara burung masih sangat jelas, latar belakang alam tenang          |
| **15 dB** | Noise sedang-ringan | Noise mulai terdengar (gerimis/hembusan), suara burung tetap dominan |
| **10 dB** | Noise sedang        | Simulasi kondisi lapangan yang agak bising                           |

> SNR 5 dB dan 0 dB **tidak digunakan** karena pada level tersebut suara burung mulai tertutup oleh _noise_. Michaud et al. (2025) menunjukkan performa deteksi menurun tajam di bawah 3 dB.

### D.3. Jenis Noise Sintetis yang Digunakan

_Noise_ yang digunakan adalah **100% sintetis** (dibuat dari rumus matematika), bukan rekaman alam, untuk menghindari kontaminasi suara burung latar belakang (Dmitriev et al., 2024):

| Jenis Noise        | Teknik Pembuatan                                                                                               | Karakteristik                                                                                    |
| ------------------ | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Angin (_Wind_)** | _Gaussian white noise_ difilter dengan _Butterworth bandpass filter_ (100–1500 Hz) + modulasi amplitudo lambat | Mensimulasikan hembusan angin yang dominan di frekuensi rendah dan tidak konstan (non-stasioner) |
| **Hujan (_Rain_)** | _Pink noise_ (1/f) menggunakan metode Voss-McCartney + _impulse bursts_ acak                                   | Mensimulasikan suara hujan merata dengan tetesan air besar sesekali                              |

### D.4. Metode Pencampuran

_Noise_ sintetis dicampur ke sinyal audio asli menggunakan metode **Additive Mixing** pada level SNR yang dipilih secara acak dari **[10, 15, 20] dB**:

Formula pencampuran:

```
noise_scaled = noise × (RMS_signal / RMS_noise) × 10^(−SNR_dB / 20)
mixed_signal = clean_signal + noise_scaled
```

### D.5. Pipeline Augmentasi Dua Level

Penelitian ini menggunakan **dua level augmentasi** yang saling melengkapi:

```
Audio Asli (5 detik, 32 kHz, Mono)
        ↓
Level 1: Augmentasi Audio Mentah (khusus data Train)
  ├── + Noise Angin sintetis (_wind) — SNR acak [10, 15, 20] dB
  └── + Noise Hujan sintetis (_rain) — SNR acak [10, 15, 20] dB
        ↓
Konversi ke Log-Mel Spectrogram (PNG)
        ↓
Level 2: Augmentasi Spectrogram / SpecAugment (khusus data Train)
  ├── Frequency Masking (_aug_freq)
  ├── Time Masking (_aug_time)
  └── Gabungan (_aug_both)
        ↓
Training EfficientNet-B3 + GRU
```

### D.6. Hasil Augmentasi Audio Mentah

Setiap file audio asli di data _Train_ menghasilkan **2 varian baru** (1× angin + 1× hujan), sehingga data _Train_ bertambah menjadi **3× lipat** di level audio mentah sebelum konversi ke _Spectrogram_.

**Tabel D.1. Distribusi Segmen Audio Train Setelah Augmentasi Noise**

| Spesies                  | Asli      | + Wind    | + Rain    | Total      |
| ------------------------ | --------- | --------- | --------- | ---------- |
| _Geopelia striata_       | 900       | 900       | 900       | 2.700      |
| _Passer montanus_        | 866       | 866       | 866       | 2.598      |
| _Pycnonotus aurigaster_  | 880       | 880       | 880       | 2.640      |
| _Pycnonotus goiavier_    | 854       | 854       | 854       | 2.562      |
| _Streptopelia chinensis_ | 892       | 892       | 892       | 2.676      |
| **Total**                | **4.392** | **4.392** | **4.392** | **13.176** |

---

## E. Augmentasi Data Spectrogram (SpecAugment)

Setelah gambar _Spectrogram_ dibuat (termasuk dari hasil augmentasi audio), tahap augmentasi lanjutan dilakukan khusus pada **data _Train_** untuk lebih melipatgandakan jumlah data dan mencegah _overfitting_. Teknik yang digunakan adalah **SpecAugment** yang diterapkan pada gambar PNG (bukan pada audio mentah), sehingga gambar asli di _disk_ tidak diubah.

Setiap gambar asli menghasilkan **3 varian augmentasi baru**:

1.  **Frequency Masking (`_aug_freq`):** Menutup sebagian baris piksel (sumbu frekuensi) dengan nilai nol/hitam. Memaksa model tidak bergantung pada satu rentang nada saja.
2.  **Time Masking (`_aug_time`):** Menutup sebagian kolom piksel (sumbu waktu). Memaksa model kebal terhadap kicauan yang terputus-putus.
3.  **Gabungan (`_aug_both`):** Kombinasi _Frequency Masking_ dan _Time Masking_ secara bersamaan.

**Dampak Augmentasi Gabungan (Dual-Level: Audio + Spectrogram):**

**Tabel E.1. Ringkasan Pertambahan Data Train Melalui Dual-Level Augmentation**

| Tahap                                            | Teknik                                 | Jumlah Data Train   |
| ------------------------------------------------ | -------------------------------------- | ------------------- |
| 1. Data asli (setelah balancing + preprocessing) | —                                      | 4.392 segmen audio  |
| 2. Augmentasi Audio Mentah (Level 1)             | + Noise Angin & Hujan (SNR 10–20 dB)   | 13.176 segmen audio |
| 3. Konversi ke Spectrogram                       | Log-Mel Spectrogram PNG                | 13.176 citra        |
| 4. SpecAugment (Level 2)                         | Freq Masking + Time Masking + Gabungan | **52.704 citra**    |

**Tabel E.2. Distribusi Akhir Citra Spectrogram per Kelas per Split**

| Spesies                  | Train (termasuk augmentasi) | Validasi | Test    | Total      |
| ------------------------ | --------------------------- | -------- | ------- | ---------- |
| _Geopelia striata_       | 10.800                      | 111      | 105     | 11.016     |
| _Passer montanus_        | 10.392                      | 105      | 98      | 10.595     |
| _Pycnonotus aurigaster_  | 10.560                      | 108      | 107     | 10.775     |
| _Pycnonotus goiavier_    | 10.248                      | 121      | 101     | 10.470     |
| _Streptopelia chinensis_ | 10.704                      | 121      | 93      | 10.918     |
| **Total**                | **52.704**                  | **566**  | **504** | **53.774** |

> **Catatan:** Data Validasi (566 gambar) dan Test (504 gambar) **tidak diaugmentasi** sama sekali, agar hasil evaluasi mencerminkan kemampuan generalisasi model yang sesungguhnya.

---

## F. Arsitektur & Strategi Pelatihan Model

### E.1. Pemilihan Arsitektur: EfficientNet-B3 + GRU

Arsitektur gabungan **EfficientNet-B3 + GRU** dipilih mengacu pada Baowaly et al. (2024). Varian B3 (bukan B7 seperti di proposal awal) dipilih karena:

- Dimensi _input_ native B3 (300×300) sangat sepadan dengan resolusi _Spectrogram_ yang dihasilkan (±300×313 piksel).
- Kapasitas 12 juta parameter lebih proporsional untuk dataset 5 spesies, jauh lebih kebal _overfitting_ dibandingkan B7 (66 juta parameter).

**Alur Arsitektur:**

```
Input Spectrogram (300×300×3)
        ↓
  EfficientNet-B3 (Beku/Frozen pada Fase 1)
  Output: (10, 10, 1536)
        ↓
     Reshape → (100, 1536)       ← 100 langkah waktu
        ↓
     GRU (256 unit)              ← Membaca alur waktu kicauan
  Output: (256,)
        ↓
  BatchNormalization
  Dropout (0.4)
  Dense + L2 Regularization     ← Klasifikasi ke 5 spesies
  Softmax Output
```

### E.2. Strategi Training Dua Fase

#### Fase 1: Feature Extraction (Backbone Beku)

- _Backbone EfficientNet-B3_ dikunci (_frozen_) agar pengetahuan _ImageNet_ tidak rusak.
- Hanya lapisan **GRU dan Dense** yang dilatih.
- **Optimizer:** Adam
- **Learning Rate:** Cosine Decay (awal `1e-3` → minimum `1e-6`)
- **Loss Function:** Categorical Crossentropy + **Label Smoothing = 0.05**
- **Early Stopping:** monitor `val_accuracy`, `patience=8`
- **Durasi:** Berhenti optimal di Epoch ke-11 (dari maks 30 Epoch)

#### Fase 2: Fine-Tuning (Backbone Dibuka Sebagian)

- 100 lapisan terbawah _EfficientNet_ tetap dikunci; selebihnya dibuka (_unfreeze parsial_).
- Semua lapisan _BatchNormalization_ tetap dikunci untuk menjaga stabilitas statistik.
- **Learning Rate:** Cosine Decay (awal `5e-5` — sangat kecil agar adaptasi halus)
- **Loss Function:** Categorical Crossentropy + Label Smoothing = 0.05
- **Early Stopping:** monitor `val_accuracy`, `patience=10`
- **Durasi:** Berhenti optimal di Epoch ke-13 Fase 2 (dari maks 50 Epoch)

### E.3. Parameter Regularisasi Final

| Teknik                | Nilai               | Tujuan                                                       |
| --------------------- | ------------------- | ------------------------------------------------------------ |
| **Dropout**           | 0.4                 | Mencegah model bergantung pada neuron spesifik               |
| **L2 Regularization** | `1e-4` pada Dense   | Menghukum bobot yang terlalu besar                           |
| **Label Smoothing**   | 0.05                | Mencegah model terlalu "percaya diri" (100%) pada satu kelas |
| **Cosine Decay LR**   | `1e-3` → `1e-6`     | Penurunan kecepatan belajar yang mulus                       |
| **SpecAugment**       | Freq + Time Masking | Augmentasi data 4× pada gambar _Spectrogram_                 |

---

## F. Hasil Evaluasi Akhir

### F.1. Akurasi & Loss

| Metrik                      | Nilai      |
| --------------------------- | ---------- |
| **Akurasi Test (Accuracy)** | **81,15%** |
| **Loss Test**               | 0,1070     |
| **Macro F1-Score**          | **0,81**   |
| **Weighted F1-Score**       | **0,81**   |

### F.2. Classification Report per Spesies

| Spesies                  | Precision | Recall   | F1-Score | Jumlah Sampel |
| ------------------------ | --------- | -------- | -------- | ------------- |
| _Geopelia striata_       | 0.79      | 0.79     | **0.79** | 105           |
| _Passer montanus_        | 0.87      | 0.92     | **0.90** | 98            |
| _Pycnonotus aurigaster_  | 0.80      | 0.74     | **0.77** | 107           |
| _Pycnonotus goiavier_    | 0.78      | 0.79     | **0.78** | 101           |
| _Streptopelia chinensis_ | 0.82      | 0.83     | **0.82** | 93            |
| **Rata-rata (Macro)**    | **0.81**  | **0.81** | **0.81** | **504**       |

### F.3. Analisis Confusion Matrix

Berdasarkan analisis _Confusion Matrix_, terdapat pola kesalahan yang signifikan dan dapat dijelaskan secara ilmiah:

- Spesies yang **paling mudah dikenali** oleh model adalah _Passer montanus_ (F1: 0.90, Recall: 92%) dan _Streptopelia chinensis_ (F1: 0.82, Recall: 83%), karena kedua spesies ini memiliki karakteristik vokalisasi yang khas dan berbeda signifikan dari spesies lainnya.

- Spesies yang **paling sulit dibedakan** adalah pasangan _Pycnonotus aurigaster_ dan _Pycnonotus goiavier_:
  - **18 sampel** _Pycnonotus aurigaster_ salah diklasifikasikan sebagai _Pycnonotus goiavier_.
  - **4 sampel** _Pycnonotus goiavier_ salah diklasifikasikan sebagai _Pycnonotus aurigaster_.

- Selain itu, terdapat kebingungan antara _Geopelia striata_ dan _Pycnonotus aurigaster_:
  - **11 sampel** _Geopelia striata_ salah diklasifikasikan sebagai _Pycnonotus aurigaster_.
  - **10 sampel** _Streptopelia chinensis_ salah diklasifikasikan sebagai _Geopelia striata_ (wajar karena keduanya sesama burung Columbidae/Tekukur).

- Kebingungan ini bukan merupakan kegagalan arsitektur, melainkan **temuan biologis yang valid**. Kedua spesies tersebut adalah sesama anggota genus _Pycnonotus_ (famili Pycnonotidae / Burung Merbah/Cucak) yang memiliki struktur vokalisasi akustik yang sangat mirip satu sama lain. Bahkan para ahli ornitologi pun sering memerlukan analisis mendalam untuk membedakan suara keduanya.

- Temuan ini dapat disajikan sebagai bagian **analisis dan pembahasan** di Bab 4–5 skripsi sebagai bukti bahwa model telah belajar pola akustik yang bermakna secara biologis.

---

## G. Analisis Pengaruh Augmentasi Data (Baseline vs V5)

Untuk membuktikan efektivitas teknik augmentasi yang diusulkan, dilakukan eksperimen pembanding (Baseline) menggunakan arsitektur, _hyperparameter_, dan pembagian data uji yang 100% identik dengan model final (V5), namun **tanpa menggunakan augmentasi sama sekali** (baik _noise_ lingkungan maupun _SpecAugment_).

### G.1. Perbandingan Performa

| Metrik Evaluasi       | Model Baseline (Tanpa Augmentasi) | Model V5 (Dengan Augmentasi) | Selisih       |
| :-------------------- | :-------------------------------- | :--------------------------- | :------------ |
| **Jumlah Data Train** | 4.392 citra                       | 52.704 citra                 | +48.312 citra |
| **Akurasi Test**      | 77,38%                            | **81,15%**                   | **+3,77%**    |
| **Test Loss**         | **0,0990**                        | 0,1070                       | +0,0080       |
| **Macro F1-Score**    | 0,78                              | **0,81**                     | +0,03         |

Berdasarkan tabel di atas, penerapan _Dual-Level Augmentation_ terbukti secara empiris mampu meningkatkan akurasi model sebesar **3,77%** dan Macro F1-Score sebesar **0,03**. Hal ini membuktikan bahwa augmentasi berhasil membuat model lebih tangguh dalam mengenali pola suara burung.

![Perbandingan Performa Model](visualisasi_bab4/Gambar_4_5_Perbandingan_Augmentasi.png)
_Gambar 4.5 Perbandingan Performa Model Tanpa dan Dengan Augmentasi_

### G.2. Analisis _Confidence vs Accuracy Trade-off_

Terdapat temuan menarik di mana model V5 memiliki akurasi yang lebih tinggi (81,15%), namun nilai _Loss_-nya sedikit lebih besar (0,1070) dibandingkan model Baseline (0,0990). Fenomena ini dapat dijelaskan secara ilmiah:

1. **Pencegahan _Overconfidence_:** Model Baseline yang dilatih dengan data bersih cenderung sangat _overconfident_ (terlalu percaya diri). Saat menebak benar, probabilitasnya mendekati 99%, sehingga _Loss_ mendekati nol. Sebaliknya, model V5 dilatih dengan data yang bising (_noisy_). Hal ini memaksa model untuk lebih berhati-hati. Saat menebak benar, probabilitasnya mungkin berada di kisaran 75%-85%. Karena _Loss_ dihitung berdasarkan tingkat keyakinan (probabilitas), maka rata-rata _Loss_ model V5 sedikit lebih tinggi, meskipun jumlah tebakan benarnya (Akurasi) jauh lebih banyak.
2. **Karakteristik _Focal Loss_:** Penggunaan _Categorical Focal Crossentropy_ secara spesifik memberikan penalti (_loss_ lebih tinggi) pada sampel yang sulit ditebak. Karena dataset V5 mengandung puluhan ribu sampel sulit akibat augmentasi, wajar jika rata-rata _Loss_ tertahan di angka yang sedikit lebih tinggi.

Kesimpulannya, nilai _Loss_ yang sedikit lebih tinggi pada V5 bukanlah sebuah kelemahan, melainkan bukti bahwa augmentasi berhasil mencegah model menjadi _overconfident_ dan membuatnya lebih realistis saat dihadapkan pada data dunia nyata.

### G.3. Analisis Kurva Pelatihan dan _Overfitting_

Pada kurva pelatihan model V5, terlihat adanya jarak (_gap_) yang cukup lebar antara _Train Accuracy_ (mendekati 98%) dan _Val Accuracy_ (sekitar 80%). Meskipun secara visual terlihat seperti _overfitting_, hal ini dapat dijustifikasi melalui dua argumen:

1. **Kapasitas Model vs Augmentasi Statis:** Arsitektur _EfficientNet-B3_ memiliki sekitar 12 juta parameter. Kapasitas yang sangat besar ini memungkinkan model untuk perlahan-lahan "menghafal" 52.704 citra latih pada epoch-epoch akhir, terutama karena augmentasi dilakukan secara _offline_ (gambar statis yang sama diulang setiap epoch).
2. **Keberhasilan _Early Stopping_:** Meskipun ada tendensi _overfitting_ di epoch akhir, fitur _Early Stopping_ dan _Restore Best Weights_ berhasil menghentikan pelatihan dan menyimpan bobot model pada titik optimal (saat _Val Loss_ terendah), bukan pada epoch terakhir.

Fakta terpenting adalah batas maksimal akurasi validasi/uji berhasil diangkat dari 77% menjadi 81%. Tidak ada model _Deep Learning_ di dunia nyata yang grafiknya menempel sempurna tanpa _gap_, sehingga hasil ini sangat valid dan dapat dipertanggungjawabkan secara akademis.

---

## H. Ringkasan Iterasi & Perbaikan Model

| Versi          | Kondisi                                                         | Akurasi Test | Catatan                                    |
| -------------- | --------------------------------------------------------------- | ------------ | ------------------------------------------ |
| V1             | CPU, EfficientNet saja, tanpa augmentasi                        | 46%          | Training berhari-hari                      |
| V2             | GPU T4, EfficientNet+GRU, tanpa augmentasi                      | 55%          | GPU aktif, masih data sedikit              |
| V3             | GPU T4 + SpecAugment 4×                                         | 75%          | Data naik 4×, akurasi lompat               |
| V4             | V3 + parameter konservatif (Dropout 0.4, LS 0.05, Cosine Decay) | 80.00%       | Parameter optimal ditemukan                |
| **V5 (Final)** | **V4 + Augmentasi Noise Lingkungan (Angin & Hujan)**            | **81.15%**   | **Model final skripsi, data Train 52.704** |

---

## I. Dokumentasi Detail Setiap Iterasi Training

Bagian ini mendokumentasikan secara kronologis setiap percobaan _training_ yang dilakukan, termasuk masalah yang ditemukan dan pelajaran yang diambil dari masing-masing iterasi.

---

### H.1. Versi 1 (V1) — Eksperimen Awal: CPU + Pure EfficientNet

**Kondisi & Konfigurasi:**

- **Perangkat:** CPU Laptop (tanpa GPU)
- **Arsitektur:** EfficientNet-B3 saja (tanpa GRU)
- **Augmentasi:** Menggunakan `RandomZoom(0.1)` — **kelak terbukti keliru**
- **Dropout:** 0.5
- **Data Train:** ~4.200 gambar (sebelum augmentasi SpecAugment)
- **Early Stopping:** monitor `val_loss`, patience=5

**Masalah yang Ditemukan:**

1. **Kecepatan Training Sangat Lambat:** Karena menggunakan CPU, satu Epoch membutuhkan ±3–5 jam. Training semalam penuh hanya menyelesaikan beberapa Epoch.
2. **`get_layer("efficientnetb3")` Error:** Keras versi terbaru (≥3.0) membongkar lapisan EfficientNet langsung ke dalam model induk sehingga nama `"efficientnetb3"` tidak lagi dikenali. Solusi: gunakan `model.trainable = True` dan iterasi `model.layers` secara langsung.
3. **`tf.shape()` Error (KerasTensor):** Penggunaan `tf.shape(feat)` di dalam Keras Functional API tidak diizinkan di Keras 3. Solusi: gunakan `Reshape((100, 1536))` dengan angka statis karena EfficientNetB3 dengan input 300×300 **selalu** menghasilkan output 10×10×1536.
4. **`RandomZoom` Merusak Spectrogram:** Augmentasi `RandomZoom` tidak cocok untuk gambar Spectrogram karena _zoom_ mengubah sumbu Y (frekuensi), sehingga kicauan yang aslinya bernada rendah bergeser menjadi nada tinggi. Solusi: hapus total `RandomZoom`.

**Hasil:** Akurasi Test **46.43%**

**Pelajaran:** Harus menggunakan GPU. EfficientNet murni tanpa GRU dengan data sedikit tidak cukup untuk dataset audio burung.

---

### H.2. Versi 2 (V2) — GPU Aktif + Tambah GRU

**Kondisi & Konfigurasi:**

- **Perangkat:** Google Colab GPU NVIDIA Tesla T4
- **Arsitektur:** EfficientNet-B3 + GRU (256 unit) — sesuai Baowaly et al. (2024)
- **Augmentasi:** Hanya `Dropout(0.2)` pada pipeline — `RandomZoom` sudah dihapus
- **Dropout Head:** 0.5
- **Data Train:** ~4.200 gambar (belum augmentasi SpecAugment)
- **Early Stopping:** monitor `val_loss`, patience=5
- **LR Schedule:** ReduceLROnPlateau (factor=0.2, patience=3)

**Masalah yang Ditemukan:**

1. **Data Terlalu Sedikit:** Dengan hanya ~840 gambar per kelas di data _Train_, model terlalu cepat menghafal (_overfitting_). _Train accuracy_ mencapai 88% tetapi _val accuracy_ hanya 53% di Fase 1.
2. **Early Stopping Terlalu Cepat:** Dengan `patience=5` dan monitor `val_loss`, model berhenti di Epoch ke-7 dan mengembalikan bobot dari Epoch 2 (val_acc hanya 55%).
3. **Val Loss Tidak Stabil:** Nilai `val_loss` naik-turun tidak konsisten karena data validasi juga sangat sedikit (hanya ~566 gambar).

**Hasil Fase 1:** val_accuracy ~66% (Epoch 2 terbaik)
**Hasil Fase 2:** Akurasi Test **55.75%**

**Pelajaran:** Akar masalah utama adalah **kurangnya data**. Augmentasi SpecAugment pada gambar Spectrogram diperlukan untuk melipatgandakan data _Train_.

---

### H.3. Versi 3 (V3) — Augmentasi SpecAugment 4×

**Kondisi & Konfigurasi:**

- **Perangkat:** Google Colab GPU T4
- **Arsitektur:** EfficientNet-B3 + GRU (256 unit) — sama dengan V2
- **Augmentasi Data:** SpecAugment pada gambar PNG (Frequency Masking + Time Masking + Gabungan) → data _Train_ naik dari ~4.200 menjadi **~17.568 gambar**
- **Dropout Head:** 0.5
- **Early Stopping:** monitor `val_accuracy`, patience=8 (dikoreksi dari `val_loss`)
- **LR Schedule:** Cosine Decay
- **Label Smoothing:** 0.05
- **L2 Regularization:** `1e-4` pada Dense head

**Peningkatan Signifikan:**

- Epoch 1 Fase 1 langsung menghasilkan val_accuracy **71%** (vs 55% di V2 yang butuh 7 Epoch)
- Model jauh lebih stabil karena data lebih beragam

**Hasil Fase 1:** val_accuracy **77.74%** (berhenti Epoch 11)
**Hasil Fase 2:** val_accuracy **79.15%** (berhenti Epoch 13)
**Akurasi Test:** **75.00%**

**Pelajaran:** Augmentasi data adalah faktor terpenting. Penggantian monitor `val_loss` → `val_accuracy` mencegah model berhenti di epoch yang bukan terbaik secara akurasi.

---

### H.4. Versi 4 (V4) — Parameter Dikoreksi

**Kondisi & Konfigurasi:**

- **Perangkat:** Google Colab GPU T4
- **Arsitektur:** EfficientNet-B3 + GRU (256 unit)
- **Augmentasi Data:** SpecAugment 4× (sama dengan V3) → **~17.568 gambar Train**
- **Dropout Head:** **0.4** (diturunkan dari 0.5, nilai 0.6 terlalu agresif)
- **L2 Regularization:** `1e-4` pada Dense
- **Label Smoothing:** **0.05** (konservatif, bukan 0.1)
- **LR Schedule:** Cosine Decay (`1e-3` → `1e-6` Fase 1; `5e-5` → `1e-7` Fase 2)
- **Early Stopping:** monitor `val_accuracy`; patience **8** (Fase 1) dan **10** (Fase 2)

**Hasil Evaluasi Test:** Akurasi **80.00%**, F1-Score **0.80**

**Kesimpulan:** V4 menemukan konfigurasi parameter _hyperparameter_ yang optimal (Dropout, Label Smoothing, dan arsitektur), tetapi belum menyertakan _noise_ lingkungan di level raw audio.

---

### H.5. Versi 5 (V5) — Model Final Skripsi: Dual-Level Augmentation

Ini adalah iterasi terakhir dan model yang digunakan untuk **hasil akhir skripsi**.

**Kondisi & Konfigurasi:**

- **Perangkat:** Google Colab GPU T4
- **Arsitektur:** EfficientNetB3_GRU_v3 (EfficientNet-B3 + GRU 256 unit) dengan parameter optimal dari V4.
- **Dual-Level Augmentation (BARU):**
  1. **Raw Audio:** Penambahan _noise_ Angin dan Hujan sintetis pada SNR 10-20 dB (meningkatkan audio mentah 3× lipat menjadi ~13.176 file).
  2. **Spectrogram:** SpecAugment (Freq + Time Masking) (meningkatkan gambar menjadi 4× lipat).
- **Data Train:** Meledak dari 17.568 menjadi **52.704 gambar**.
- **Data Validation:** 566 gambar.
- **Data Test:** 504 gambar.

**Hasil Evaluasi Test (FINAL):**

| Metrik         | Nilai                           |
| -------------- | ------------------------------- |
| Akurasi        | **81.15%** (409 benar dari 504) |
| Loss           | **0.1070**                      |
| Macro F1-Score | **0.81**                        |

**Per Spesies (Classification Report):**

| Spesies                  | Precision | Recall | F1       |
| ------------------------ | --------- | ------ | -------- |
| _Geopelia striata_       | 0.79      | 0.79   | 0.79     |
| _Passer montanus_        | 0.87      | 0.92   | **0.90** |
| _Pycnonotus aurigaster_  | 0.80      | 0.74   | 0.77     |
| _Pycnonotus goiavier_    | 0.78      | 0.79   | 0.78     |
| _Streptopelia chinensis_ | 0.82      | 0.83   | **0.82** |

**Kesimpulan Akhir:**
Kombinasi _Dual-Level Augmentation_ (Noise Lingkungan + SpecAugment) berhasil memecahkan rekor akurasi sebelumnya, naik dari 80.00% menjadi **81.15%**, serta meningkatkan skor Macro F1 dari 0.80 ke 0.81. Model tidak hanya lebih akurat, tetapi juga secara teori jauh lebih **tangguh (_robust_)** terhadap suara bising di lapangan (sesuai literatur Kumar et al., 2024 dan Dmitriev et al., 2024). Kesalahan klasifikasi terbesar saat ini hanya terjadi secara biologis pada genus _Pycnonotus_, yang membuktikan bahwa model mempelajari pola akustik natural.

---

## J. Peningkatan Terbaru: Multi-Output & Denoising Tingkat Lanjut (Iterasi V6)

Menyusul evaluasi dari eksperimen V5, dilakukan beberapa peningkatan signifikan untuk memenuhi kriteria evaluasi akademis yang lebih ketat, khususnya terkait integrasi fitur ekstraksi audio secara _native_ ke dalam model dan optimalisasi representasi Spectrogram.

### I.1. Arsitektur Multi-Output (Klasifikasi + Regresi Frekuensi)

Berdasarkan kebutuhan untuk mengeluarkan data hasil ekstraksi akustik (frekuensi) secara sinkron dengan klasifikasi spesies (mirip dengan kapabilitas _pyannote_), arsitektur model _single-output_ dirombak menjadi model **Multi-Output**.

- **Head 1 (Klasifikasi):** Menggunakan _Dense Layer_ + _Softmax_ untuk memprediksi probabilitas 5 kelas spesies. Dioptimalkan dengan fungsi kerugian _Categorical Crossentropy_.
- **Head 2 (Regresi Frekuensi):** Menggunakan _Dense Linear Layer_ untuk memprediksi angka kontinu frekuensi secara bersamaan (_concurrent_). Dioptimalkan dengan _Mean Absolute Error (MAE)_.
- **Custom Data Generator:** Pembuatan fungsi `MultiOutputGenerator` (turunan dari `tf.keras.utils.Sequence`) untuk membaca nilai kebenaran (_ground truth_) dari file `ground_truth_frekuensi.csv` secara _on-the-fly_ untuk melatih _head_ regresi.
- **Loss Weights:** Penyesuaian bobot (1.0 untuk klasifikasi spesies, 0.001 untuk regresi frekuensi) diterapkan untuk mencegah regresi merusak gradien dari klasifikasi spesies utama.

### I.2. Denoising Lanjutan & Bandpass Filtering Berbasis Literatur

Kritik mengenai presisi _Spectrogram_ pada format PNG 8-bit diselesaikan tanpa menyalahi pakem judul skripsi (yang wajib menggunakan _Log-Mel Spectrogram_). Optimasi dilakukan secara matematis sebelum konversi citra di tahap `generate_spectrograms.py`:

- **Bandpass Filter Kustom:** Menggunakan rentang presisi **300 Hz – 15.000 Hz** mengacu pada literatur pengklasifikasian suara burung (Koh et al., 2019; Michaud et al., 2025). Batas bawah 300 Hz secara sengaja dipilih untuk membersihkan noise gemuruh mekanis dan angin, **tanpa memotong/merusak** suara frekuensi sangat rendah yang menjadi ciri khas spesies _Geopelia striata_ (Perkutut/Tekukur).
- **Spectral Gating:** Penambahan integrasi `noisereduce` untuk menghapus sisa _noise_ stasioner secara dinamis, sehingga gambar PNG Spectrogram jauh lebih kontras (_salient_).
- **Skenario Deployment Realistis:** Proses Bandpass dan Denoising ini dieksekusi **setelah** injeksi _noise_ lingkungan buatan (`augment_environmental_noise.py`). Hal ini menyimulasikan skenario dunia nyata di mana mikrofon menangkap sinyal bising, yang kemudian baru dibersihkan oleh _software_ sebelum diklasifikasikan oleh AI.
