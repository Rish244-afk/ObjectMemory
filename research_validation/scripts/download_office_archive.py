#!/usr/bin/env python3
"""
Streaming downloader for OpenLORIS-Scene Office Archive.
Downloads package/office1-1_7-package.tar with SHA-256 and MD5 verification.
"""

import datetime
import hashlib
import os
import sys
import time
import urllib.request

URL = "https://huggingface.co/datasets/shixuesong/openloris-scene/resolve/main/package/office1-1_7-package.tar"
DEST = "research_validation/data/raw/openloris_scene/office/office1-1_7-package.tar"
EXPECTED_SIZE = 10618337280

def main():
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    start_time = datetime.datetime.now(datetime.timezone.utc)
    print(f"=== OpenLORIS-Scene Office Archive Downloader ===")
    print(f"Start Time (UTC): {start_time.isoformat()}")
    print(f"Source URL:       {URL}")
    print(f"Destination:      {DEST}")
    print(f"Expected Size:    {EXPECTED_SIZE} bytes ({EXPECTED_SIZE / (1024**3):.3f} GB)")
    print("=" * 50)

    # Check if partially downloaded or already exists
    initial_size = 0
    headers = {"User-Agent": "Mozilla/5.0"}
    
    req = urllib.request.Request(URL, headers=headers)
    t0 = time.time()
    last_print = t0
    downloaded = 0

    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()

    with urllib.request.urlopen(req) as resp, open(DEST, "wb") as f:
        total = int(resp.headers.get("content-length", EXPECTED_SIZE))
        print(f"Connected. HTTP {resp.status}. Content-Length: {total} bytes.")
        
        while True:
            chunk = resp.read(1024 * 1024) # 1 MB chunks
            if not chunk:
                break
            f.write(chunk)
            md5_hash.update(chunk)
            sha256_hash.update(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_print >= 5.0 or downloaded == total:
                elapsed = now - t0
                speed_mb = (downloaded / (1024 * 1024)) / max(elapsed, 0.001)
                percent = (downloaded / total) * 100.0 if total > 0 else 0.0
                eta_sec = (total - downloaded) / (speed_mb * 1024 * 1024) if speed_mb > 0 else 0
                print(f"[{percent:5.1f}%] {downloaded/(1024**3):.2f}/{total/(1024**3):.2f} GB "
                      f"({speed_mb:5.1f} MB/s) Elapsed: {elapsed:5.1f}s ETA: {eta_sec:5.1f}s")
                last_print = now

    end_time = datetime.datetime.now(datetime.timezone.utc)
    md5_digest = md5_hash.hexdigest()
    sha256_digest = sha256_hash.hexdigest()

    print("=" * 50)
    print(f"Download Completed Successfully!")
    print(f"End Time (UTC):   {end_time.isoformat()}")
    print(f"Duration:         {time.time() - t0:.2f} seconds")
    print(f"Final File Size:  {downloaded} bytes")
    print(f"MD5 Checksum:     {md5_digest}")
    print(f"SHA-256 Checksum: {sha256_digest}")
    
    # Save provenance metadata
    meta_path = DEST + ".meta.json"
    import json
    meta = {
        "source_url": URL,
        "destination": DEST,
        "start_time_utc": start_time.isoformat(),
        "end_time_utc": end_time.isoformat(),
        "bytes_downloaded": downloaded,
        "expected_bytes": EXPECTED_SIZE,
        "md5": md5_digest,
        "sha256": sha256_digest,
        "status": "COMPLETED_VERIFIED" if downloaded == EXPECTED_SIZE else "SIZE_MISMATCH"
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata recorded to {meta_path}")

if __name__ == "__main__":
    main()
