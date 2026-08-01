import os
import shutil
import warnings
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import random

warnings.filterwarnings("ignore", category=UserWarning)

# ================= KONFIGURASI UTAMA =================
INPUT_DIR = Path(__file__).parent / "dataset_duration_balanced"
OUTPUT_DIR = Path(__file__).parent / "dataset_final"

# 1. Format Audio Sesuai Baowaly/BirdCLEF
TARGET_SR = 32000
CHUNK_LENGTH_SEC = 5.0
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_LENGTH_SEC)

# 2. Salient Filtering (Threshold Suara Aktif)
# Nilai RMS ini menentukan seberapa "keras" suara untuk dianggap bukan hening.
# Jika chunk dibuang terlalu banyak, turunkan angka ini (misal 0.001).
# Jika chunk hening masih lolos, naikkan angka ini (misal 0.005).
RMS_THRESHOLD = 0.002  

# 3. Target Durasi Split (File-level Split)
# Ingat total durasi per kelas adalah 5250 detik.
TRAIN_TARGET = 4200.0  # 80%
VAL_TARGET = 525.0     # 10%
# Sisanya otomatis masuk Test (10%)
# =====================================================

def normalize_audio(y):
    """Peak Amplitude Normalization (Normalisasi skala ke rentang -1.0 s/d 1.0)"""
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        return y / max_amp
    return y

def is_salient(y):
    """Mendeteksi apakah ada sinyal suara aktif di dalam chunk"""
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = np.mean(rms)
    return mean_rms > RMS_THRESHOLD

def split_and_process():
    if not INPUT_DIR.exists():
        print(f"Folder {INPUT_DIR} tidak ditemukan! Harap pastikan dataset sudah di-balance.")
        return

    # Bersihkan folder output jika sudah ada agar chunk lama tidak bercampur
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)
        
    species_folders = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    
    print("=" * 70)
    print(" AUDIO PIPELINE: Splitting, 32kHz Mono, Norm, Salient Chunking (5s)")
    print("=" * 70)
    
    for species_dir in species_folders:
        species_name = species_dir.name
        
        # Buat sub-folder spesies di dalam train, val, dan test
        for split in ['train', 'val', 'test']:
            (OUTPUT_DIR / split / species_name).mkdir(exist_ok=True)
            
        audio_files = list(species_dir.glob("*.mp3")) + list(species_dir.glob("*.wav"))
        
        # Acak urutan file agar pendistribusian ke Train/Val/Test bersifat adil (unbiased)
        random.seed(42) # Seed agar hasil acaknya bisa diulangi jika di run ulang
        random.shuffle(audio_files) 
        
        print(f"\nMemproses Spesies: {species_name}")
        pbar = tqdm(total=len(audio_files), desc="Processing files")
        
        current_dur = 0.0
        stats = {'train': 0, 'val': 0, 'test': 0}
        
        for file_idx, f in enumerate(audio_files, 1):
            try:
                # LANGKAH 1: Load file -> Paksa ke Mono -> Resample ke 32.000 Hz
                y, sr = librosa.load(f, sr=TARGET_SR, mono=True)
                
                # Durasi asli file ini dalam detik
                dur = len(y) / sr
                
                # LANGKAH 2: Penentuan Split di Level File (File-level split)
                # Seluruh potongan dari file ini akan masuk ke 1 keranjang yang sama!
                if current_dur < TRAIN_TARGET:
                    target_split = 'train'
                elif current_dur < (TRAIN_TARGET + VAL_TARGET):
                    target_split = 'val'
                else:
                    target_split = 'test'
                    
                dest_folder = OUTPUT_DIR / target_split / species_name
                
                # LANGKAH 3: Normalisasi Amplitudo
                y = normalize_audio(y)
                
                # LANGKAH 4: Pemotongan 5 Detik (Chunking)
                total_samples = len(y)
                chunk_idx = 1
                
                for start in range(0, total_samples, CHUNK_SAMPLES):
                    end = start + CHUNK_SAMPLES
                    chunk = y[start:end]
                    
                    # LANGKAH 5: Zero-Padding untuk chunk terakhir yang durasinya kurang dari 5 detik
                    if len(chunk) < CHUNK_SAMPLES:
                        pad_length = CHUNK_SAMPLES - len(chunk)
                        chunk = np.pad(chunk, (0, pad_length), mode='constant')
                        
                    # LANGKAH 6: Salient Filtering (Buang chunk hening/angin doang)
                    if is_salient(chunk):
                        out_name = f"{species_name}_f{file_idx}_c{chunk_idx}.wav"
                        out_path = dest_folder / out_name
                        sf.write(str(out_path), chunk, TARGET_SR)
                        stats[target_split] += 1
                        
                    chunk_idx += 1
                
                current_dur += dur
                pbar.update(1)
                
            except Exception as e:
                pbar.update(1)
                continue
            
        pbar.close()
        
        print(f"  - Train : {stats['train']} chunk (Target ideal: ~840)")
        print(f"  - Valid : {stats['val']} chunk (Target ideal: ~105)")
        print(f"  - Test  : {stats['test']} chunk (Target ideal: ~105)")
        print("  *(Perbedaan angka terjadi karena chunk hening dibuang & pemotongan file di perbatasan split)*")

    print(f"\n[SELESAI] Fase 1 selesai! Semua file (.wav standar) tersimpan di:\n{OUTPUT_DIR}")

if __name__ == '__main__':
    split_and_process()
