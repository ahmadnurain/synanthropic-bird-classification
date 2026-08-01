import os
import warnings
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import noisereduce as nr
import scipy.signal
import matplotlib.pyplot as plt
import gc

warnings.filterwarnings("ignore", category=UserWarning)

# ================= KONFIGURASI =================
INPUT_DIR = Path(__file__).parent / "dataset_final"
# Output ke folder khusus baseline agar tidak bercampur dengan V5
OUTPUT_DIR = Path(__file__).parent / "dataset_baseline" / "dataset_spectrograms"

# Parameter Spectrogram (100% IDENTIK dengan V5)
TARGET_SR = 32000
N_MELS = 128
FMIN = 300
FMAX = 15000
N_FFT = 2048
HOP_LENGTH = 512
LOWCUT_HZ = 300
HIGHCUT_HZ = 15000

# Kata kunci augmentasi yang HARUS DIABAIKAN
AUG_KEYWORDS = ['_wind', '_rain']
# ===============================================

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    y = scipy.signal.lfilter(b, a, data)
    return y

def process_wav_to_spectrogram(wav_path, out_path):
    y, sr = librosa.load(wav_path, sr=TARGET_SR)
    y_filtered = bandpass_filter(y, LOWCUT_HZ, HIGHCUT_HZ, sr)
    y_clean = nr.reduce_noise(y=y_filtered, sr=sr, stationary=True, prop_decrease=0.8)
    
    S = librosa.feature.melspectrogram(
        y=y_clean, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.imsave(out_path, S_dB, cmap='magma', origin='lower')
    
    # PEMBERSIHAN MEMORI AGRESIF
    del y, y_filtered, y_clean, S, S_dB
    plt.close('all')
    gc.collect()

def generate_baseline_spectrograms():
    if not INPUT_DIR.exists():
        print(f"Folder {INPUT_DIR} tidak ditemukan!")
        return
        
    print("=" * 70)
    print(" MEMBUAT SPECTROGRAM BASELINE (TANPA AUGMENTASI)")
    print("=" * 70)
    
    # Hitung total file asli (tanpa augmentasi)
    total_files = 0
    for wav_file in INPUT_DIR.rglob("*.wav"):
        if not any(kw in wav_file.name.lower() for kw in AUG_KEYWORDS):
            total_files += 1
            
    pbar = tqdm(total=total_files, desc="Generate Baseline PNG")
    
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    for split in ['train', 'val', 'test']:
        split_dir = INPUT_DIR / split
        if not split_dir.exists(): continue
            
        for species_dir in split_dir.iterdir():
            if not species_dir.is_dir(): continue
                
            out_species_dir = OUTPUT_DIR / split / species_dir.name
            out_species_dir.mkdir(parents=True, exist_ok=True)
            
            for wav_file in species_dir.glob("*.wav"):
                # SKIP file augmentasi di folder train
                if split == 'train' and any(kw in wav_file.name.lower() for kw in AUG_KEYWORDS):
                    continue
                    
                out_name = wav_file.stem + ".png"
                out_path = out_species_dir / out_name
                
                if not out_path.exists():
                    try:
                        process_wav_to_spectrogram(wav_file, out_path)
                    except Exception as e:
                        tqdm.write(f"[ERROR] {wav_file.name}: {e}")
                        
                stats[split] += 1
                pbar.update(1)
                
    pbar.close()
    print(f"\n[SELESAI] Spectrogram Baseline siap!")
    print(f"  - Train : {stats['train']} citra (Harus 4.392)")
    print(f"  - Val   : {stats['val']} citra (Harus 566)")
    print(f"  - Test  : {stats['test']} citra (Harus 504)")
    print(f"\nSilakan ZIP folder '{OUTPUT_DIR.parent}' dan upload ke Google Drive.")

if __name__ == '__main__':
    generate_baseline_spectrograms()
