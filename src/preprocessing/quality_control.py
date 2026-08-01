import os
import shutil
import warnings
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

# Menyembunyikan warning dari librosa (terkait format audio tertentu)
warnings.filterwarnings("ignore", category=UserWarning)

# ================= KONDISI DAN PARAMETER SELEKSI =================
# Anda bisa mengubah nilai-nilai ini sesuai kebutuhan skripsi Anda
CONFIG = {
    'min_duration': 3.0,             # minimal durasi file mentah
    'max_duration': 1000.0,           # maksimal durasi file mentah (misal 5 menit)
    'valid_sample_rates': [22050, 32000, 44100, 48000],
    'target_sample_rate': 32000,     # kalau mau konsisten dengan paper Baowaly
    'max_silence_ratio': 0.70,       # jangan terlalu ketat di awal
    'min_rms': 0.003,                # cukup longgar untuk rekaman lapangan
    'max_clipping_ratio': 0.02,      # toleransi clipping ringan
    'min_snr_db': 3.0,               # longgar dulu supaya data tidak banyak gugur
    'min_active_segments': 1,
    'top_db_split': 30
}

# Folder konfigurasi
INPUT_DIR = Path(__file__).parent / "dataset"
OUTPUT_CSV = Path(__file__).parent / "audio_qc_report.csv"
OUTPUT_PASS_DIR = Path(__file__).parent / "dataset_filtered"

# =================================================================

def estimate_snr(y, non_silent_intervals):
    """
    Mengestimasi SNR (Signal-to-Noise Ratio) secara sederhana.
    Membandingkan energi (mean squared amplitude) dari bagian bersuara (signal)
    dengan bagian yang hening (noise background).
    """
    if len(non_silent_intervals) == 0:
        return 0.0

    # Buat mask (penanda) untuk bagian yang hening
    silent_mask = np.ones(len(y), dtype=bool)
    for start, end in non_silent_intervals:
        silent_mask[start:end] = False
    
    # Jika nyaris tidak ada bagian hening (audio full suara), anggap SNR tinggi
    if np.sum(silent_mask) < len(y) * 0.05:
        return 50.0

    noise_power = np.mean(y[silent_mask] ** 2)
    signal_power = np.mean(y[~silent_mask] ** 2)

    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
        return float(snr_db)
    return 50.0

def process_audio(filepath):
    """
    Mengekstrak fitur teknis dari satu file audio dan melakukan evaluasi Quality Control.
    """
    results = {
        'filepath': str(filepath),
        'filename': filepath.name,
        'species': filepath.parent.name,
        'duration_sec': 0.0,
        'sample_rate': 0,
        'silence_ratio': 0.0,
        'rms': 0.0,
        'peak_amp': 0.0,
        'clipping_ratio': 0.0,
        'estimated_snr_db': 0.0,
        'active_segments': 0,
        'status': 'REJECT',
        'notes': ''
    }
    notes = []

    try:
        # Load audio (sr=None agar mempertahankan sample rate asli)
        y, sr = librosa.load(filepath, sr=None)
        
        # 1. Cek Durasi & SR
        duration = librosa.get_duration(y=y, sr=sr)
        results['duration_sec'] = round(duration, 2)
        results['sample_rate'] = sr
        
        if duration == 0:
            results['notes'] = "Audio kosong (0 detik)"
            return results
            
        # 2. Cek Silence & Active Segments
        # Memisahkan bagian audio bersuara dengan yang hening (threshold top_db)
        non_silent_intervals = librosa.effects.split(y, top_db=CONFIG['top_db_split'])
        results['active_segments'] = len(non_silent_intervals)
        
        active_samples = sum([end - start for start, end in non_silent_intervals])
        silence_ratio = 1.0 - (active_samples / len(y))
        results['silence_ratio'] = round(silence_ratio, 3)

        # 3. RMS Energy & Peak
        results['rms'] = float(np.mean(librosa.feature.rms(y=y)))
        results['peak_amp'] = float(np.max(np.abs(y)))
        
        # 4. Clipping Ratio (audio terlalu keras/pecah)
        # Menghitung persentase sampel yang amplitudonya mendekati maksimal (0.99)
        results['clipping_ratio'] = float(np.sum(np.abs(y) >= 0.99) / len(y))
        
        # 5. SNR
        results['estimated_snr_db'] = round(estimate_snr(y, non_silent_intervals), 2)

        # ================= EVALUASI KELAYAKAN =================
        # Mengecek satu per satu kriteria untuk menentukan REJECT, REVIEW, atau PASS
        
        if duration < CONFIG['min_duration']:
            notes.append(f"Terlalu pendek (<{CONFIG['min_duration']}s)")
        if duration > CONFIG['max_duration']:
            notes.append(f"Terlalu panjang (>{CONFIG['max_duration']}s)")
        # Uncomment jika SR asli XC sangat bermasalah (biasanya untuk preprocessing ML, SR akan diseragamkan ulang, jadi ini bisa diabaikan)
        # if sr not in CONFIG['valid_sample_rates']:
        #     notes.append(f"SR tidak standar ({sr}Hz)")
        if silence_ratio > CONFIG['max_silence_ratio']:
            notes.append(f"Terlalu banyak silence ({silence_ratio*100:.1f}%)")
        if results['rms'] < CONFIG['min_rms']:
            notes.append("Energi suara terlalu lemah (RMS kecil)")
        if results['clipping_ratio'] > CONFIG['max_clipping_ratio']:
            notes.append("Audio terlalu pecah/clipping")
        if results['active_segments'] < CONFIG['min_active_segments']:
            notes.append("Tidak terdeteksi kicauan/suara")
            
        # Penetapan Status Akhir
        if len(notes) > 0:
            # Jika ada catatan error, periksa apakah ini layak REVIEW atau murni REJECT
            # Anda bisa menyesuaikan aturan ini. Misalnya, terlalu panjang bisa masuk REVIEW (karena bisa dipotong manual).
            if "Terlalu panjang" in notes and len(notes) == 1:
                results['status'] = 'REVIEW'
            elif results['estimated_snr_db'] < CONFIG['min_snr_db'] and results['estimated_snr_db'] > 0:
                 # Jika SNR jelek tapi syarat lain lolos, beri REVIEW
                 results['status'] = 'REVIEW'
            else:
                results['status'] = 'REJECT'
        else:
            # Jika tidak ada note dan SNR baik
            if results['estimated_snr_db'] >= CONFIG['min_snr_db']:
                results['status'] = 'PASS'
            else:
                notes.append("SNR (kualitas jernih) rendah")
                results['status'] = 'REVIEW'
                
        results['notes'] = " | ".join(notes) if notes else "OK"

    except Exception as e:
        results['status'] = 'REJECT'
        results['notes'] = f"Gagal membaca file: {str(e)}"

    return results

def main():
    if not INPUT_DIR.exists():
        print(f"Folder {INPUT_DIR} tidak ditemukan!")
        return

    # Ambil semua file audio (MP3, WAV, dll) dari subfolder
    audio_files = list(INPUT_DIR.rglob("*.mp3")) + list(INPUT_DIR.rglob("*.wav"))
    
    if len(audio_files) == 0:
        print("Tidak ada file audio ditemukan dalam folder dataset.")
        return

    print("="*60)
    print(f"  AUDIO QUALITY CONTROL SCRIPT")
    print(f"  Total file terdeteksi: {len(audio_files)}")
    print("="*60)

    results_data = []

    # Memproses file dengan progress bar (tqdm)
    for filepath in tqdm(audio_files, desc="Mengevaluasi Audio", unit="file"):
        res = process_audio(filepath)
        results_data.append(res)

    # Simpan hasil ke Dataframe dan CSV
    df = pd.DataFrame(results_data)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n[INFO] Laporan evaluasi disimpan di: {OUTPUT_CSV}")

    # Rekapitulasi
    pass_count = len(df[df['status'] == 'PASS'])
    review_count = len(df[df['status'] == 'REVIEW'])
    reject_count = len(df[df['status'] == 'REJECT'])

    print(f"\n  REKAP HASIL:")
    print(f"    ✓ PASS   : {pass_count}")
    print(f"    ● REVIEW : {review_count}")
    print(f"    ✗ REJECT : {reject_count}")

    # Menyalin file PASS ke folder baru (Opsional sesuai request)
    if pass_count > 0:
        print(f"\n[INFO] Menyalin {pass_count} file PASS ke {OUTPUT_PASS_DIR}...")
        OUTPUT_PASS_DIR.mkdir(parents=True, exist_ok=True)
        
        for _, row in tqdm(df[df['status'] == 'PASS'].iterrows(), total=pass_count, desc="Copying PASS files"):
            src_path = Path(row['filepath'])
            species = row['species']
            filename = row['filename']
            
            # Buat subfolder spesies di folder tujuan
            dest_species_dir = OUTPUT_PASS_DIR / species
            dest_species_dir.mkdir(parents=True, exist_ok=True)
            
            dest_path = dest_species_dir / filename
            shutil.copy2(src_path, dest_path)
            
        print("[INFO] Penyalinan selesai!")

if __name__ == '__main__':
    main()
