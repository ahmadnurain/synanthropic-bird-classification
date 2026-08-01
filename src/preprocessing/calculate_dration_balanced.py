import os
import warnings
import librosa
from pathlib import Path
from tqdm import tqdm

# Mengabaikan peringatan dari librosa
warnings.filterwarnings("ignore", category=UserWarning)

def format_duration(seconds):
    """Fungsi bantuan untuk mengubah detik menjadi format Jam, Menit, Detik"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours} Jam {minutes} Menit {secs} Detik"
    else:
        return f"{minutes} Menit {secs} Detik"

def calculate_folder_duration(folder_path):
    species_folders = [f for f in folder_path.iterdir() if f.is_dir()]
    
    if not species_folders:
        print(f"Folder {folder_path.name} kosong.")
        return
        
    total_dataset_duration = 0.0
    
    for species_dir in species_folders:
        species_name = species_dir.name
        audio_files = list(species_dir.glob("*.mp3")) + list(species_dir.glob("*.wav"))
        
        species_duration = 0.0
        
        # tqdm dengan leave=False agar progress bar hilang setelah selesai (biar rapi)
        for f in tqdm(audio_files, desc=f"Menghitung {species_name}", leave=False):
            try:
                # Baca file secara akurat untuk memotong error perhitungan VBR header MP3
                y, sr = librosa.load(f, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                species_duration += duration
            except Exception as e:
                # Jika ada file corrupt yang terlewat, abaikan saja
                pass
                
        total_dataset_duration += species_duration
        formatted_time = format_duration(species_duration)
        
        print(f"- {species_name.ljust(25)} : {formatted_time}  (dari {len(audio_files)} file)")
        
    # Print Grand Total
    formatted_total = format_duration(total_dataset_duration)
    print("-" * 70)
    print(f"TOTAL KESELURUHAN DURASI : {formatted_total}")
    print("-" * 70)
    print("\n")

def main():
    print("\n" + "="*70)
    print("KALKULATOR DURASI REKAMAN DATASET (UNTUK LAPORAN SKRIPSI)")
    print("="*70 + "\n")
    
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "dataset_duration_balanced"
    # filtered_dir = base_dir / "dataset_filtered"
    # balanced_dir = base_dir / "dataset_balanced"
    
    if raw_dir.exists():
        print("=== 1. DATASET balndced (Sebelum Di Sergamakan) ===")
        calculate_folder_duration(raw_dir)
        
    # if filtered_dir.exists():
    #     print("=== 2. DATASET FILTERED (Data Asli Lolos QC, Sebelum Augmentasi) ===")
    #     calculate_folder_duration(filtered_dir)
    # else:
    #     print("[WARN] Folder dataset_filtered tidak ditemukan.")
        
    # if balanced_dir.exists():
    #     print("=== 3. DATASET BALANCED (Data Setelah Undersampling & Oversampling) ===")
    #     calculate_folder_duration(balanced_dir)

if __name__ == '__main__':
    main()
