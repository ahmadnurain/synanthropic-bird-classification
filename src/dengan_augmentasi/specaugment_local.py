"""
specaugment_local.py — Augmentasi SpecAugment pada Spektrogram PCEN (Lokal)
==========================================================================
Menerapkan SpecAugment (Frequency Masking + Time Masking) pada data TRAIN
sehingga di Colab langsung bisa training tanpa perlu augmentasi lagi.

Dari setiap gambar asli, dihasilkan 3 gambar augmentasi:
  - _aug_freq.png  → Frequency Mask saja
  - _aug_time.png  → Time Mask saja
  - _aug_both.png  → Frequency + Time Mask

Total data train akan menjadi 4x lipat (1 asli + 3 augmentasi).
"""

import os
import random
import cv2
from pathlib import Path
from tqdm import tqdm

# ================= KONFIGURASI =================
SPECTROGRAM_DIR = Path(__file__).parent / "dataset_spectrograms"
TRAIN_DIR = SPECTROGRAM_DIR / "train"
AUG_DONE_FLAG = SPECTROGRAM_DIR / "augmentation_done.flag"
# ===============================================


def freq_mask(img, max_ratio=0.15):
    """Menutupi sebagian band frekuensi (sumbu Y) dengan warna hitam."""
    aug = img.copy()
    h = aug.shape[0]
    f = random.randint(1, max(1, int(h * max_ratio)))
    f0 = random.randint(0, h - f)
    aug[f0:f0+f, :] = 0
    return aug


def time_mask(img, max_ratio=0.15):
    """Menutupi sebagian frame waktu (sumbu X) dengan warna hitam."""
    aug = img.copy()
    w = aug.shape[1]
    t = random.randint(1, max(1, int(w * max_ratio)))
    t0 = random.randint(0, w - t)
    aug[:, t0:t0+t] = 0
    return aug


def run_specaugment():
    if not TRAIN_DIR.exists():
        print(f"Folder {TRAIN_DIR} tidak ditemukan!")
        print("Jalankan generate_spectrograms.py terlebih dahulu.")
        return

    if AUG_DONE_FLAG.exists():
        print("SpecAugment sudah pernah dilakukan sebelumnya.")
        print(f"Hapus file {AUG_DONE_FLAG} jika ingin mengulang.")
        return

    print("=" * 70)
    print(" SPECAUGMENT: Frequency Masking + Time Masking (Lokal)")
    print("=" * 70)

    random.seed(42)  # Seed agar hasil bisa direproduksi
    total_added = 0

    species_dirs = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]

    for species_dir in species_dirs:
        # Hanya proses file asli (tanpa _aug di nama)
        original_files = [
            f for f in species_dir.glob("*.png")
            if "_aug" not in f.stem
        ]

        print(f"\n[{species_dir.name}] {len(original_files)} gambar asli")
        
        for fpath in tqdm(original_files, desc=f"  Augmentasi", leave=False):
            img = cv2.imread(str(fpath))
            if img is None:
                continue

            base = fpath.stem

            # 1. Frequency Mask saja
            cv2.imwrite(
                str(species_dir / f"{base}_aug_freq.png"),
                freq_mask(img)
            )

            # 2. Time Mask saja
            cv2.imwrite(
                str(species_dir / f"{base}_aug_time.png"),
                time_mask(img)
            )

            # 3. Frequency + Time Mask (keduanya)
            cv2.imwrite(
                str(species_dir / f"{base}_aug_both.png"),
                time_mask(freq_mask(img))
            )

            total_added += 3

    # Tandai bahwa augmentasi sudah selesai
    AUG_DONE_FLAG.touch()

    print(f"\n[SELESAI] {total_added} gambar augmentasi ditambahkan ke data train!")

    # Tampilkan statistik akhir
    print("\nStatistik Dataset:")
    for split in ['train', 'val', 'test']:
        split_dir = SPECTROGRAM_DIR / split
        if not split_dir.exists():
            continue
        total = sum(
            len(list(d.glob("*.png")))
            for d in split_dir.iterdir() if d.is_dir()
        )
        print(f"  {split}: {total} gambar")


if __name__ == '__main__':
    run_specaugment()
