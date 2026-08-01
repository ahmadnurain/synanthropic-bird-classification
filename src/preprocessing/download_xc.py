"""
Download Audio Dataset dari Xeno-canto API v3
==============================================
Script untuk skripsi: Klasifikasi Suara Burung Sinantropik
menggunakan EfficientNet pada Citra Mel-Spectrogram.

Target: 6 spesies burung sinantropik kawasan kaki Gunung Ciremai
Filter: Tanpa batas (semua kualitas dan durasi) untuk memaksimalkan dataset.

Catatan:
- Streptopelia chinensis sekarang diklasifikasikan sebagai Spilopelia chinensis
  di database Xeno-canto (IOC taxonomy). Keduanya spesies yang sama.
- Folder output menggunakan nama ilmiah standar untuk konsistensi.
"""

import asyncio
import aiohttp
import aiofiles
import os
import json
import sys
import time
from pathlib import Path

# ================= CONFIGURATION =================
API_KEY = "c168b28df8de8a1520b94460b8108477db2e6d67"
BASE_URL = "https://xeno-canto.org/api/3/recordings"

# Folder output — akan dibuat otomatis
OUTPUT_DIR = Path(__file__).parent / "dataset"

# Concurrent downloads per species
MAX_CONCURRENT = 5

# Rate limit delay between API pages (seconds)
PAGE_DELAY = 1.0

# Results per page (max 500)
PER_PAGE = 500
# ==================================================

# 6 Spesies target sesuai proposal skripsi
# Format: (nama_ilmiah_xc, nama_folder, nama_lokal)
SPECIES_LIST = [
    ("Passer montanus",           "Passer montanus",           "Burung Gereja"),
    ("Pycnonotus aurigaster",     "Pycnonotus aurigaster",     "Kutilang"),
    ("Pycnonotus goiavier",       "Pycnonotus goiavier",       "Trucukan"),
    ("Lonchura leucogastroides",  "Lonchura leucogastroides",  "Bondol Jawa"),
    ("Geopelia striata",          "Geopelia striata",           "Perkutut"),
    # Streptopelia chinensis = Spilopelia chinensis di XC (IOC taxonomy)
    ("Spilopelia chinensis",      "Streptopelia chinensis",     "Tekukur"),
]


async def fetch_metadata(session: aiohttp.ClientSession, species_name: str) -> list:
    """
    Fetch semua recording metadata untuk satu spesies.
    Filter: Tanpa filter kualitas dan durasi agar data maksimal.
    """
    query = f'sp:"{species_name}"'
    page = 1
    all_recordings = []

    while True:
        params = {
            'query': query,
            'key': API_KEY,
            'per_page': PER_PAGE,
            'page': page,
        }

        try:
            async with session.get(BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 401:
                    print(f"  [ERROR] 401 Unauthorized — API Key tidak valid!")
                    return []
                if resp.status == 429:
                    print(f"  [WARN] Rate limited. Menunggu 10 detik...")
                    await asyncio.sleep(10)
                    continue
                if resp.status != 200:
                    print(f"  [ERROR] HTTP {resp.status} untuk {species_name}")
                    break

                data = await resp.json()
        except asyncio.TimeoutError:
            print(f"  [WARN] Timeout halaman {page}, mencoba ulang...")
            await asyncio.sleep(3)
            continue
        except Exception as e:
            print(f"  [ERROR] Gagal fetch metadata: {e}")
            break

        recordings = data.get('recordings', [])
        all_recordings.extend(recordings)

        num_pages = int(data.get('numPages', 1))
        num_recordings = int(data.get('numRecordings', 0))
        print(f"  Halaman {page}/{num_pages} — {len(recordings)} rekaman (total kumulatif: {len(all_recordings)})")

        if page >= num_pages:
            break
        page += 1
        await asyncio.sleep(PAGE_DELAY)

    return all_recordings


async def download_one(
    session: aiohttp.ClientSession,
    recording: dict,
    folder_path: Path,
    semaphore: asyncio.Semaphore,
    stats: dict,
):
    """Download satu file audio."""
    track_id = recording.get('id', 'unknown')
    file_url = recording.get('file', '')

    if not file_url:
        stats['skipped'] += 1
        return

    # Fix URL protocol jika diawali '//'
    if file_url.startswith('//'):
        file_url = 'https:' + file_url

    file_path = folder_path / f"XC{track_id}.mp3"

    # Skip jika sudah ada
    if file_path.exists():
        stats['exists'] += 1
        return

    async with semaphore:
        retries = 3
        for attempt in range(1, retries + 1):
            try:
                async with session.get(
                    file_url,
                    timeout=aiohttp.ClientTimeout(total=120),
                    allow_redirects=True,
                ) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        async with aiofiles.open(str(file_path), 'wb') as f:
                            await f.write(content)
                        stats['downloaded'] += 1
                        # Progress indicator setiap 10 file
                        total = stats['downloaded'] + stats['exists']
                        if stats['downloaded'] % 10 == 0:
                            print(f"    ✓ {stats['downloaded']} file baru didownload...")
                        return
                    elif resp.status == 429:
                        wait = 5 * attempt
                        print(f"    [429] Rate limited XC{track_id}, tunggu {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        print(f"    [WARN] HTTP {resp.status} untuk XC{track_id}")
                        stats['failed'] += 1
                        return
            except asyncio.TimeoutError:
                if attempt < retries:
                    await asyncio.sleep(2 * attempt)
                else:
                    stats['failed'] += 1
            except Exception as e:
                stats['failed'] += 1
                if attempt == retries:
                    print(f"    [ERROR] XC{track_id}: {e}")
                return


async def process_species(session: aiohttp.ClientSession, species_xc: str, folder_name: str, nama_lokal: str):
    """Proses satu spesies: fetch metadata + download semua file."""
    print(f"\n{'='*60}")
    print(f"  {nama_lokal} ({species_xc})")
    print(f"{'='*60}")

    # Fetch metadata
    recordings = await fetch_metadata(session, species_xc)
    if not recordings:
        print(f"  Tidak ada rekaman ditemukan untuk {species_xc}!")
        return

    print(f"  Total ditemukan: {len(recordings)} rekaman (Semua kualitas & durasi)")

    # Buat folder
    folder_path = OUTPUT_DIR / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    # Save metadata JSON untuk referensi
    meta_path = folder_path / "_metadata.json"
    meta_summary = []
    for r in recordings:
        meta_summary.append({
            'id': r.get('id'),
            'en': r.get('en'),
            'cnt': r.get('cnt'),
            'loc': r.get('loc'),
            'q': r.get('q'),
            'length': r.get('length'),
            'type': r.get('type'),
            'rec': r.get('rec'),
            'date': r.get('date'),
        })
    async with aiofiles.open(str(meta_path), 'w', encoding='utf-8') as f:
        await f.write(json.dumps(meta_summary, indent=2, ensure_ascii=False))

    # Download
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    stats = {'downloaded': 0, 'exists': 0, 'skipped': 0, 'failed': 0}

    tasks = [download_one(session, rec, folder_path, sem, stats) for rec in recordings]
    await asyncio.gather(*tasks)

    print(f"\n  Hasil {nama_lokal}:")
    print(f"    ✓ Baru didownload : {stats['downloaded']}")
    print(f"    ● Sudah ada       : {stats['exists']}")
    print(f"    ✗ Gagal           : {stats['failed']}")
    print(f"    - Dilewati (no URL): {stats['skipped']}")

    # Hitung total file di folder
    total_files = len(list(folder_path.glob("XC*.mp3")))
    print(f"    📁 Total file MP3 : {total_files}")


async def main():
    print("=" * 60)
    print("  DOWNLOAD DATASET SUARA BURUNG SINANTROPIK")
    print("  Sumber: Xeno-canto API v3")
    print("  Filter: Semua kualitas dan durasi (Dimaksimalkan)")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Jumlah spesies: {len(SPECIES_LIST)}")
    print("=" * 60)

    # Buat output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    connector = aiohttp.TCPConnector(limit=10, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        for species_xc, folder_name, nama_lokal in SPECIES_LIST:
            await process_species(session, species_xc, folder_name, nama_lokal)

    # Summary akhir
    print(f"\n{'='*60}")
    print("  RINGKASAN DATASET")
    print(f"{'='*60}")
    for _, folder_name, nama_lokal in SPECIES_LIST:
        folder = OUTPUT_DIR / folder_name
        if folder.exists():
            count = len(list(folder.glob("XC*.mp3")))
            print(f"  {nama_lokal:<20} ({folder_name}): {count} file")
        else:
            print(f"  {nama_lokal:<20} ({folder_name}): 0 file")
    print(f"\n  Semua selesai! Cek folder: {OUTPUT_DIR}")


if __name__ == '__main__':
    print(f"Python {sys.version}")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDihentikan oleh user (Ctrl+C)")
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
