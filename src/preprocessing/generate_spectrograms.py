import os
import warnings
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import noisereduce as nr  # Untuk Spectral Gating / Denoising
import scipy.signal       # Untuk Bandpass Filter

# Abaikan warning dari librosa
warnings.filterwarnings("ignore", category=UserWarning)

# ================= KONFIGURASI =================
INPUT_DIR = Path(__file__).parent / "dataset_final"
OUTPUT_DIR = Path(__file__).parent / "dataset_spectrograms"

# Parameter Spectrogram
TARGET_SR = 32000
N_MELS = 128
FMIN = 300
FMAX = 15000

# Parameter standar analisis frekuensi Fast-Fourier Transform (FFT)
N_FFT = 2048
HOP_LENGTH = 512

# Parameter Denoising (Bandpass Filter)
LOWCUT_HZ = 300
HIGHCUT_HZ = 15000
# ===============================================


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Filter untuk mengisolasi frekuensi kicauan burung."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    y = scipy.signal.lfilter(b, a, data)
    return y


def normalize_to_uint8(arr):
    """Normalisasi array ke rentang 0-255 (uint8) untuk disimpan sebagai channel PNG."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    normalized = (arr - min_val) / (max_val - min_val) * 255.0
    return normalized.astype(np.uint8)


def process_wav_to_spectrogram(wav_path, out_path):
    """
    Membaca audio dan mengubahnya menjadi gambar Log-Mel Spectrogram.
    """
    # 1. Load Audio
    y, sr = librosa.load(wav_path, sr=TARGET_SR)
    
    # --- TAHAP DENOISING & ISOLASI SUARA ---
    # 1A. Bandpass Filter (Hanya ambil frekuensi spesifik burung)
    y_filtered = bandpass_filter(y, LOWCUT_HZ, HIGHCUT_HZ, sr)
    
    # 1B. Spectral Gating (Meredam noise background yang konsisten/statis)
    y_clean = nr.reduce_noise(y=y_filtered, sr=sr, stationary=True, prop_decrease=0.8)
    # -------------------------------------------------------
    
    # 2. Ekstraksi Mel-Spectrogram
    S = librosa.feature.melspectrogram(
        y=y_clean,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )
    
    # 3. Log-Mel (Konversi ke Desibel)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # 4. Simpan langsung sebagai gambar PNG dengan colormap 'magma'
    import matplotlib.pyplot as plt
    plt.imsave(out_path, S_dB, cmap='magma', origin='lower')


def generate_all_spectrograms():
    if not INPUT_DIR.exists():
        print(f"Folder {INPUT_DIR} tidak ditemukan! Pastikan audio pipeline sudah dijalankan.")
        return
        
    print("=" * 70)
    print(" EKSTRAKSI LOG-MEL SPECTROGRAM (V6)")
    print("=" * 70)
    print(f"Parameter  : Mels={N_MELS}, Freq={FMIN}-{FMAX} Hz")
    print(f"Feature    : Log-Mel Spectrogram (dB)")
    print(f"Input Data : {INPUT_DIR}")
    print(f"Output     : {OUTPUT_DIR}\n")
    
    # Hitung total file untuk tampilan progress bar
    total_files = sum(1 for _ in INPUT_DIR.rglob("*.wav"))
    
    if total_files == 0:
        print("Tidak ada file audio di dalam dataset_final.")
        return
        
    pbar = tqdm(total=total_files, desc="Generate Log-Mel PNG")
    
    # Memproses ketiga split (train, val, test)
    for split in ['train', 'val', 'test']:
        split_dir = INPUT_DIR / split
        if not split_dir.exists(): continue
            
        for species_dir in split_dir.iterdir():
            if not species_dir.is_dir(): continue
                
            # Buat folder struktur yang sama persis di output
            out_species_dir = OUTPUT_DIR / split / species_dir.name
            out_species_dir.mkdir(parents=True, exist_ok=True)
            
            for wav_file in species_dir.glob("*.wav"):
                # Ganti ekstensi .wav jadi .png
                out_name = wav_file.stem + ".png"
                out_path = out_species_dir / out_name
                
                # Fitur Resume: Hanya proses jika file gambar belum dibuat
                if not out_path.exists():
                    try:
                        process_wav_to_spectrogram(wav_file, out_path)
                    except Exception as e:
                        tqdm.write(f"[ERROR] {wav_file.name}: {e}")
                        
                pbar.update(1)
                
    pbar.close()
    print(f"\n[SELESAI] Spektrogram Log-Mel (V6) siap dilatih!")
    print(f"Cek gambar hasilnya di: {OUTPUT_DIR}")


if __name__ == '__main__':
    generate_all_spectrograms()
