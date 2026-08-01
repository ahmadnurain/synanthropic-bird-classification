"""
Augmentasi Audio dengan Noise Lingkungan (Angin & Hujan)
=========================================================
Skrip ini menambahkan noise lingkungan sintetis MURNI (angin dan hujan)
ke segmen audio burung pada data TRAIN untuk meningkatkan robustness model
terhadap kondisi rekaman lapangan yang tidak ideal.

Noise yang digunakan adalah SINTETIS (dibuat dari rumus matematika),
bukan rekaman alam, untuk menghindari kontaminasi suara burung latar
belakang (Dmitriev et al., 2024).

Referensi Ilmiah:
  1. Kumar, V. et al. (2024). "Improving learning-based birdsong classification
     by utilizing combined audio augmentation strategies."
     Ecological Informatics.
     -> Membuktikan kombinasi strategi augmentasi audio (termasuk simulated
        pink noise dan background noise injection) menghasilkan performa
        terbaik untuk klasifikasi kicauan burung.

  2. Dmitriev, A. et al. (2024). "Methods for Training Convolutional Neural
     Networks to Identify Bird Sounds."
     -> Noise hujan dan angin bersifat non-stasioner dan rekaman alam
        sering mengandung suara burung latar belakang, sehingga noise
        sintetis lebih aman digunakan.

  3. Hong, J. et al. (2023). "Acoustic Bird Species Recognition at BirdCLEF 2023."
     -> Menggunakan GaussianNoise dan PinkNoise pada audio mentah sebelum
        konversi ke Mel-spectrogram sebagai teknik augmentasi standar.

  4. Alzahra, S. et al. (2019). "Bird sounds classification by combining PNCC
     and robust Mel-log filter bank features."
     -> Memberikan dasar angka SNR eksplisit pada klasifikasi suara burung:
        augmentasi pada 4 level SNR (20 dB, 10 dB, 5 dB, 0 dB).
        Menjadi acuan pemilihan rentang SNR 10-20 dB pada penelitian ini.

  5. Michaud, A. et al. (2025). "Acoustic detection of a nocturnal bird with
     deep learning: the challenge of low signal-to-noise ratio."
     -> Menunjukkan performa model deteksi burung menurun tajam ketika
        SNR di bawah 3 dB. Menjadi dasar untuk TIDAK menggunakan SNR 0 dB.

Setting SNR yang dipilih (10, 15, 20 dB):
  - 20 dB = noise ringan (kondisi alam tenang)
  - 15 dB = noise sedang-ringan (ada hembusan angin/gerimis)
  - 10 dB = noise sedang (kondisi lapangan yang agak bising)

Penggunaan:
  python augment_environmental_noise.py
"""

import os
import warnings
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from scipy.signal import butter, lfilter

warnings.filterwarnings("ignore", category=UserWarning)

# ================= KONFIGURASI =================
# Folder input: hasil chunking 5 detik (hanya folder TRAIN)
DATASET_DIR = Path(__file__).parent / "dataset_final"
TRAIN_DIR = DATASET_DIR / "train"

# Parameter Audio
TARGET_SR = 32000
CHUNK_DURATION = 5.0
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_DURATION)

# Parameter SNR (Signal-to-Noise Ratio)
# Berdasarkan Alzahra et al. (2019) yang menggunakan SNR 20, 10, 5, 0 dB.
# Dipilih 3 level AMAN (10-20 dB) karena Michaud et al. (2025)
# menunjukkan performa menurun tajam di bawah 3 dB.
SNR_LEVELS_DB = [10, 15, 20]  # dB
# ===============================================


def generate_wind_noise(n_samples, sr=32000):
    """
    Menghasilkan noise sintetis MURNI yang menyerupai suara angin.
    
    Teknik: Gaussian white noise difilter dengan Butterworth bandpass filter
    pada rentang frekuensi rendah (100-1500 Hz) untuk mensimulasikan
    karakteristik spektral hembusan angin yang dominan di frekuensi rendah.
    Ditambahkan modulasi amplitudo lambat untuk efek non-stasioner
    (kadang kencang kadang pelan), sesuai karakteristik angin asli
    (Dmitriev et al., 2024).
    
    TIDAK menggunakan rekaman alam untuk menghindari kontaminasi
    suara burung latar belakang.
    """
    # Generate white noise
    noise = np.random.randn(n_samples).astype(np.float32)
    
    # Desain bandpass filter (100 Hz - 1500 Hz) untuk karakter angin
    nyquist = sr / 2
    low = 100 / nyquist
    high = 1500 / nyquist
    
    # Clamp agar tidak melebihi batas Nyquist
    low = max(low, 0.001)
    high = min(high, 0.999)
    
    b, a = butter(N=4, Wn=[low, high], btype='band')
    wind = lfilter(b, a, noise)
    
    # Modulasi amplitudo lambat untuk efek "hembusan" non-stasioner
    mod_freq = np.random.uniform(0.3, 1.5)  # frekuensi modulasi (Hz)
    t = np.linspace(0, n_samples / sr, n_samples)
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t + np.random.uniform(0, 2 * np.pi))
    wind = wind * modulation
    
    return wind.astype(np.float32)


def generate_rain_noise(n_samples, sr=32000):
    """
    Menghasilkan noise sintetis MURNI yang menyerupai suara hujan.
    
    Teknik: Kombinasi dari:
    1. Pink noise (1/f noise) sebagai latar belakang hujan merata
    2. Impulse bursts acak sebagai simulasi tetesan air besar
    
    Pink noise memiliki energi yang menurun secara proporsional terhadap
    frekuensi (spektrum 1/f), yang sangat mirip dengan karakteristik
    spektral hujan asli (Kumar et al., 2024).
    
    TIDAK menggunakan rekaman alam untuk menghindari kontaminasi
    suara burung latar belakang.
    """
    # Generate pink noise (1/f noise) menggunakan metode Voss-McCartney
    n_rows = n_samples
    n_cols = 16  # Jumlah sumber noise (semakin banyak = semakin smooth)
    
    # Inisialisasi array
    array = np.random.randn(n_rows, n_cols).astype(np.float32)
    
    # Mask untuk update bertahap (simulasi 1/f)
    for col in range(n_cols):
        period = 2 ** col
        mask = np.zeros(n_rows, dtype=bool)
        mask[::period] = True
        
        hold_values = np.random.randn(int(np.ceil(n_rows / period))).astype(np.float32)
        idx = 0
        current_val = hold_values[0]
        for row in range(n_rows):
            if mask[row] and idx < len(hold_values):
                current_val = hold_values[idx]
                idx += 1
            array[row, col] = current_val
    
    pink = np.sum(array, axis=1)
    
    # Normalize pink noise
    max_val = np.max(np.abs(pink))
    if max_val > 0:
        pink = pink / max_val
    
    # Tambahkan impulse bursts (simulasi tetesan air besar)
    n_drops = np.random.randint(50, 200)
    for _ in range(n_drops):
        pos = np.random.randint(0, n_samples)
        drop_len = np.random.randint(50, 300)
        if pos + drop_len < n_samples:
            drop_amplitude = np.random.uniform(0.1, 0.4)
            drop = drop_amplitude * np.random.randn(drop_len).astype(np.float32)
            # Envelope tetesan (cepat naik, lambat turun)
            envelope = np.exp(-np.linspace(0, 5, drop_len))
            pink[pos:pos + drop_len] += drop * envelope
    
    return pink.astype(np.float32)


def mix_at_snr(clean_signal, noise_signal, snr_db):
    """
    Mencampur sinyal bersih dengan noise pada level SNR tertentu (dalam dB).
    
    SNR (Signal-to-Noise Ratio) = perbandingan kekuatan suara burung vs noise.
    Semakin besar SNR, suara burung semakin jelas.
    
    Level yang digunakan (Alzahra et al., 2019; Michaud et al., 2025):
    - SNR 20 dB = noise ringan (kondisi alam tenang)
    - SNR 15 dB = noise sedang-ringan (gerimis/hembusan)
    - SNR 10 dB = noise sedang (lapangan agak bising)
    
    Formula: noise_scaled = noise * (rms_signal / rms_noise) * 10^(-snr_db/20)
    """
    # Hitung RMS (energi rata-rata) dari sinyal bersih
    rms_signal = np.sqrt(np.mean(clean_signal ** 2))
    rms_noise = np.sqrt(np.mean(noise_signal ** 2))
    
    if rms_noise == 0 or rms_signal == 0:
        return clean_signal
    
    # Skalakan noise agar sesuai dengan SNR yang diinginkan
    target_rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise_scaled = noise_signal * (target_rms_noise / rms_noise)
    
    # Campur (additive mixing)
    mixed = clean_signal + noise_scaled
    
    # Clipping prevention: normalisasi jika melebihi batas
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val
    
    return mixed.astype(np.float32)


def augment_train_data():
    """
    Proses utama: iterasi seluruh file WAV di folder train,
    lalu buat 2 varian augmentasi per file (1x angin, 1x hujan)
    pada SNR acak dari [10, 15, 20] dB.
    """
    if not TRAIN_DIR.exists():
        print(f"[ERROR] Folder {TRAIN_DIR} tidak ditemukan!")
        print("Harap jalankan audio_pipeline.py terlebih dahulu.")
        return
    
    species_folders = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
    
    print("=" * 70)
    print(" AUGMENTASI AUDIO: Noise Lingkungan Sintetis (Angin & Hujan)")
    print("=" * 70)
    print(f" SNR Levels    : {SNR_LEVELS_DB} dB")
    print(f"   20 dB = noise ringan (alam tenang)")
    print(f"   15 dB = noise sedang-ringan (gerimis/hembusan)")
    print(f"   10 dB = noise sedang (lapangan agak bising)")
    print(f" Jenis Noise   : Wind (Angin sintetis) + Rain (Hujan sintetis)")
    print(f" Output        : 2 file baru per file asli (_wind, _rain)")
    print(f" Target        : Hanya data TRAIN (Val/Test tidak disentuh)")
    print("=" * 70)
    
    total_created = 0
    
    for species_dir in species_folders:
        species_name = species_dir.name
        
        # Ambil hanya file WAV asli (bukan hasil augmentasi sebelumnya)
        wav_files = [f for f in species_dir.glob("*.wav") 
                     if "_wind" not in f.stem and "_rain" not in f.stem]
        
        print(f"\n[{species_name}] {len(wav_files)} file asli ditemukan...")
        
        species_created = 0
        
        for wav_file in tqdm(wav_files, desc=f"  Augmenting {species_name}"):
            try:
                y, sr = librosa.load(wav_file, sr=TARGET_SR, mono=True)
                
                # Pastikan panjang konsisten (5 detik)
                if len(y) < CHUNK_SAMPLES:
                    y = np.pad(y, (0, CHUNK_SAMPLES - len(y)), mode='constant')
                elif len(y) > CHUNK_SAMPLES:
                    y = y[:CHUNK_SAMPLES]
                
                n_samples = len(y)
                
                # Pilih SNR acak dari [10, 15, 20] dB
                snr_wind = np.random.choice(SNR_LEVELS_DB)
                snr_rain = np.random.choice(SNR_LEVELS_DB)
                
                # === AUGMENTASI 1: Tambah Noise Angin Sintetis ===
                wind_noise = generate_wind_noise(n_samples, sr)
                y_wind = mix_at_snr(y, wind_noise, snr_wind)
                
                wind_name = wav_file.stem + f"_wind.wav"
                sf.write(str(species_dir / wind_name), y_wind, sr)
                
                # === AUGMENTASI 2: Tambah Noise Hujan Sintetis ===
                rain_noise = generate_rain_noise(n_samples, sr)
                y_rain = mix_at_snr(y, rain_noise, snr_rain)
                
                rain_name = wav_file.stem + f"_rain.wav"
                sf.write(str(species_dir / rain_name), y_rain, sr)
                
                species_created += 2
                
            except Exception as e:
                continue
        
        total_created += species_created
        print(f"  -> {species_created} file augmentasi baru ({species_name})")
    
    print(f"\n{'=' * 70}")
    print(f" SELESAI! Total {total_created} file audio augmentasi baru ditambahkan.")
    print(f" Lokasi: {TRAIN_DIR}")
    print(f"{'=' * 70}")
    print(f"\n[LANGKAH SELANJUTNYA]")
    print(f" Jalankan generate_spectrograms.py untuk mengonversi hasil")
    print(f" augmentasi ini ke Log-Mel Spectrogram.")


if __name__ == "__main__":
    augment_train_data()
