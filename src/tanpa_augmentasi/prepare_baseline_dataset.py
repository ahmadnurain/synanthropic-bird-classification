"""
Script untuk menyiapkan dataset_baseline.zip
Menyalin HANYA gambar spectrogram ASLI (tanpa augmentasi) ke folder baru,
lalu mengompresnya menjadi ZIP untuk diupload ke Google Drive.

File yang DIBUANG (tidak disalin):
  - *_wind*.png     (augmentasi noise angin)
  - *_rain*.png     (augmentasi noise hujan)
  - *_aug_freq*.png (frequency masking)
  - *_aug_time*.png (time masking)
  - *_aug_both*.png (gabungan masking)
"""
import os
import shutil
import zipfile

# === KONFIGURASI ===
SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_spectrograms")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_baseline")
ZIP_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_baseline.zip")

# Kata kunci augmentasi yang harus dibuang dari data TRAIN
AUG_KEYWORDS = ['_wind', '_rain', '_aug_freq', '_aug_time', '_aug_both']

def is_augmented(filename):
    """Cek apakah file adalah hasil augmentasi berdasarkan nama file."""
    fn_lower = filename.lower()
    return any(kw in fn_lower for kw in AUG_KEYWORDS)

def main():
    print("=" * 60)
    print("MENYIAPKAN DATASET BASELINE (TANPA AUGMENTASI)")
    print("=" * 60)

    if not os.path.exists(SOURCE_DIR):
        print(f"[ERROR] Folder sumber tidak ditemukan: {SOURCE_DIR}")
        print("Pastikan folder 'dataset_spectrograms/' ada di direktori yang sama.")
        return

    # Bersihkan folder output jika sudah ada
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for split in ['train', 'val', 'test']:
        src_split = os.path.join(SOURCE_DIR, split)
        dst_split = os.path.join(OUTPUT_DIR, "dataset_spectrograms", split)

        if not os.path.exists(src_split):
            print(f"[WARN] Folder {src_split} tidak ditemukan, dilewati.")
            continue

        total_copied = 0
        total_skipped = 0

        for species in sorted(os.listdir(src_split)):
            src_species = os.path.join(src_split, species)
            if not os.path.isdir(src_species):
                continue

            dst_species = os.path.join(dst_split, species)
            os.makedirs(dst_species, exist_ok=True)

            for filename in os.listdir(src_species):
                if not filename.lower().endswith('.png'):
                    continue

                # Untuk data TRAIN: buang file augmentasi
                # Untuk data VAL dan TEST: salin semua (tidak ada augmentasi)
                if split == 'train' and is_augmented(filename):
                    total_skipped += 1
                    continue

                src_file = os.path.join(src_species, filename)
                dst_file = os.path.join(dst_species, filename)
                shutil.copy2(src_file, dst_file)
                total_copied += 1

        print(f"\n[{split.upper()}]")
        print(f"  Disalin  : {total_copied} file")
        if split == 'train':
            print(f"  Dibuang  : {total_skipped} file (augmentasi)")

    # Buat ZIP untuk diupload ke Google Drive
    print(f"\nMembuat ZIP: {ZIP_OUTPUT}")
    with zipfile.ZipFile(ZIP_OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, OUTPUT_DIR)
                zipf.write(abs_path, rel_path)

    zip_size_mb = os.path.getsize(ZIP_OUTPUT) / (1024 * 1024)
    print(f"\n[OK] dataset_baseline.zip berhasil dibuat ({zip_size_mb:.1f} MB)")
    print("Upload file ini ke Google Drive/Skripsi/ lalu jalankan notebook baseline.")

if __name__ == "__main__":
    main()
