import os
import shutil
import random
import warnings
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ================= KONFIGURASI =================
INPUT_DIR = Path(__file__).parent / "dataset"
OUTPUT_DIR = Path(__file__).parent / "dataset_duration_balanced"

# Target: 1 Jam 27 Menit 30 Detik = (1 * 3600) + (27 * 60) + 30 = 5250 Detik
TARGET_DURATION = 5250.0  
EXCLUDE_SPECIES = ["Lonchura leucogastroides"] # Bondol Jawa dihapus
# ===============================================

def augment_audio(y, sr):
    """Fungsi augmentasi: merubah pitch, merubah speed, atau menambah noise"""
    methods = ['pitch_shift', 'time_stretch', 'add_noise']
    method = random.choice(methods)
    
    if method == 'pitch_shift':
        n_steps = random.uniform(-2, 2)
        y_aug = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    elif method == 'time_stretch':
        rate = random.uniform(0.8, 1.2)
        y_aug = librosa.effects.time_stretch(y=y, rate=rate)
    else: 
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y_aug = y + noise_amp * np.random.normal(size=y.shape[0])
        
    return y_aug

def process_species(species_dir, dest_dir):
    audio_files = list(species_dir.glob("*.mp3")) + list(species_dir.glob("*.wav"))
    random.shuffle(audio_files) # Acak file agar variatif
    
    current_duration = 0.0
    
    print(f"\nMemproses {species_dir.name}...")
    pbar = tqdm(total=TARGET_DURATION, desc="  Durasi", unit="dtk", bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f} dtk")
    
    # TAHAP 1: Mengambil file asli sampai target durasi tercapai (Undersampling)
    for f in audio_files:
        if current_duration >= TARGET_DURATION:
            break
            
        try:
            y, sr = librosa.load(f, sr=None)
            dur = librosa.get_duration(y=y, sr=sr)
            
            if current_duration + dur <= TARGET_DURATION:
                # Jika muat utuh, salin filenya langsung
                shutil.copy2(f, dest_dir / f.name)
                current_duration += dur
                pbar.update(dur)
            else:
                # Jika kelebihan, POTONG file bagian belakangnya agar totalnya PAS dengan target
                needed = TARGET_DURATION - current_duration
                samples_needed = int(needed * sr)
                y_trimmed = y[:samples_needed] # Potong array audio
                
                new_name = f.stem + "_trimmed.wav"
                sf.write(str(dest_dir / new_name), y_trimmed, sr)
                current_duration += needed
                pbar.update(needed)
                break
        except Exception as e:
            # Lewati file rusak
            pass
            
    # TAHAP 2: Jika file asli sudah habis tapi target belum tercapai (Oversampling/Augmentasi)
    # Ini khusus terjadi pada spesies yang total durasi aslinya kurang dari 1 Jam 27 Menit (misal: Kutilang)
    aug_counter = 1
    while current_duration < TARGET_DURATION:
        f = random.choice(audio_files) # Pilih acak file aslinya
        try:
            y, sr = librosa.load(f, sr=None)
            y_aug = augment_audio(y, sr)
            dur_aug = librosa.get_duration(y=y_aug, sr=sr)
            
            if current_duration + dur_aug <= TARGET_DURATION:
                # Simpan full augmentasi
                new_name = f"aug_{aug_counter}_{f.stem}.wav"
                sf.write(str(dest_dir / new_name), y_aug, sr)
                current_duration += dur_aug
                pbar.update(dur_aug)
            else:
                # Potong hasil augmentasi agar pas
                needed = TARGET_DURATION - current_duration
                samples_needed = int(needed * sr)
                y_trimmed = y_aug[:samples_needed]
                
                new_name = f"aug_{aug_counter}_{f.stem}_trimmed.wav"
                sf.write(str(dest_dir / new_name), y_trimmed, sr)
                current_duration += needed
                pbar.update(needed)
                break
            aug_counter += 1
        except Exception as e:
            pass

    pbar.close()
    
def main():
    if not INPUT_DIR.exists():
        print(f"Folder {INPUT_DIR} tidak ditemukan!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PENYERAGAMAN DURASI DATASET")
    print("Target: 1 Jam 27 Menit 30 Detik (5250 Detik) Per Spesies")
    print("=" * 70)

    species_folders = [f for f in INPUT_DIR.iterdir() if f.is_dir()]
    
    for species_dir in species_folders:
        if species_dir.name in EXCLUDE_SPECIES:
            print(f"\n[SKIP] Melewati {species_dir.name} karena telah dihapus dari daftar.")
            continue
            
        dest_dir = OUTPUT_DIR / species_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        process_species(species_dir, dest_dir)
        
    print(f"\n[SELESAI] Data dengan durasi seragam telah disimpan di:")
    print(f"{OUTPUT_DIR}")

if __name__ == '__main__':
    main()
