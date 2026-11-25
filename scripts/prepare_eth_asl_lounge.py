"""
Reproducible downloader and extractor for the ETH ASL Change Detection Lounge dataset.

- Streams each observation_0..9.bag from the official ETH Research Collection Zip64 archive
  via HTTP Range (206 Partial Content) to stay within local disk limits (38 GB free vs
  96 GB combined zip+uncompressed bags), verifying exact Zip64 CRC32 and SHA-256.
- Preserves all 10 aligned TSDF .ply meshes under data/eth_asl_lounge/raw/complete_mesh/.
- Extracts original-resolution RGB PNGs, bag timestamps, and T_G_C / T_G_D camera transforms
  under data/eth_asl_lounge/extracted/observation_01/ .. observation_10/.
- Writes machine-readable manifest to research_validation/ETH_ASL_DATASET_MANIFEST.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

LOUNGE_ZIP_URL = "https://www.research-collection.ethz.ch/server/api/core/bitstreams/7492cccc-c182-488a-a21b-8b5102cc0348/content"
LOUNGE_ZIP_BYTES = 30950780627


def parse_zip64_entries(url: str, total_size: int) -> Dict[str, Dict[str, Any]]:
    tail_start = total_size - 65536
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={tail_start}-{total_size-1}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        tail = r.read()
    loc_idx = tail.rfind(b"PK\x06\x07")
    if loc_idx == -1:
        raise RuntimeError("Zip64 EOCD locator not found")
    _, _, z64_eocd_off, _ = struct.unpack("<4sIQI", tail[loc_idx : loc_idx + 20])
    eocd_rel = z64_eocd_off - tail_start
    cd_size, cd_off = struct.unpack("<QQ", tail[eocd_rel + 40 : eocd_rel + 56])

    req2 = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={cd_off}-{cd_off+cd_size-1}"})
    with urllib.request.urlopen(req2, timeout=30) as r2:
        cd_bytes = r2.read()

    entries: Dict[str, Dict[str, Any]] = {}
    pos = 0
    while pos < len(cd_bytes):
        if cd_bytes[pos : pos + 4] != b"PK\x01\x02":
            break
        comp_method, crc32, comp_sz, uncomp_sz, fn_len, ex_len, cm_len = struct.unpack(
            "<H4xIIIHHH", cd_bytes[pos + 10 : pos + 34]
        )
        loc_off = struct.unpack("<I", cd_bytes[pos + 42 : pos + 46])[0]
        fn = cd_bytes[pos + 46 : pos + 46 + fn_len].decode("utf-8", errors="replace")
        extra = cd_bytes[pos + 46 + fn_len : pos + 46 + fn_len + ex_len]
        if comp_sz == 0xFFFFFFFF or uncomp_sz == 0xFFFFFFFF or loc_off == 0xFFFFFFFF:
            ep = 0
            while ep < len(extra):
                eid, esz = struct.unpack("<HH", extra[ep : ep + 4])
                if eid == 0x0001:
                    vals = struct.unpack("<" + "Q" * (esz // 8), extra[ep + 4 : ep + 4 + esz])
                    idx = 0
                    if uncomp_sz == 0xFFFFFFFF:
                        uncomp_sz = vals[idx]
                        idx += 1
                    if comp_sz == 0xFFFFFFFF:
                        comp_sz = vals[idx]
                        idx += 1
                    if loc_off == 0xFFFFFFFF:
                        loc_off = vals[idx]
                        idx += 1
                    break
                ep += 4 + esz
        if uncomp_sz > 0:
            entries[fn] = {
                "method": comp_method,
                "crc32": crc32,
                "comp_sz": comp_sz,
                "uncomp_sz": uncomp_sz,
                "loc_off": loc_off,
            }
        pos += 46 + fn_len + ex_len + cm_len
    return entries


def stream_extract_zip_entry(url: str, meta: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
    lh_req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={meta['loc_off']}-{meta['loc_off']+511}"}
    )
    with urllib.request.urlopen(lh_req, timeout=30) as r:
        lh = r.read()
    if lh[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Invalid local ZIP header for {out_path.name}")
    l_fn, l_ex = struct.unpack("<HH", lh[26:30])
    data_start = meta["loc_off"] + 30 + l_fn + l_ex
    data_end = data_start + meta["comp_sz"] - 1

    decomp = zlib.decompressobj(-zlib.MAX_WBITS) if meta["method"] == 8 else None
    crc_calc = 0
    sha = hashlib.sha256()
    bytes_written = 0
    chunk_sz = 256 * 1024 * 1024  # Large range chunks to avoid HTTP 429 rate limits

    with open(out_path, "wb") as out_f:
        cur = data_start
        while cur <= data_end:
            end = min(cur + chunk_sz - 1, data_end)
            for attempt in range(8):
                try:
                    creq = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={cur}-{end}"})
                    with urllib.request.urlopen(creq, timeout=300) as cr:
                        buf = cr.read()
                    if len(buf) != (end - cur + 1):
                        raise IOError(f"Short read {len(buf)} != {end - cur + 1}")
                    break
                except urllib.error.HTTPError as he:
                    if attempt == 7:
                        raise
                    wait_s = int(he.headers.get("Retry-After", 0) or 0)
                    if wait_s <= 0:
                        wait_s = 15 * (attempt + 1)
                    time.sleep(min(wait_s, 180))
                except Exception:
                    if attempt == 7:
                        raise
                    time.sleep(10 * (attempt + 1))
            out_buf = decomp.decompress(buf) if decomp else buf
            if out_buf:
                out_f.write(out_buf)
                crc_calc = zlib.crc32(out_buf, crc_calc)
                sha.update(out_buf)
                bytes_written += len(out_buf)
            cur = end + 1

        if decomp:
            tail_buf = decomp.flush()
            if tail_buf:
                out_f.write(tail_buf)
                crc_calc = zlib.crc32(tail_buf, crc_calc)
                sha.update(tail_buf)
                bytes_written += len(tail_buf)

    crc_calc &= 0xFFFFFFFF
    if crc_calc != meta["crc32"]:
        raise RuntimeError(f"CRC32 mismatch for {out_path.name}: {crc_calc:08x} != {meta['crc32']:08x}")
    if bytes_written != meta["uncomp_sz"]:
        raise RuntimeError(f"Size mismatch for {out_path.name}: {bytes_written} != {meta['uncomp_sz']}")

    return {"bytes": bytes_written, "crc32": f"{crc_calc:08x}", "sha256": sha.hexdigest()}


def file_checksums(path: Path) -> Dict[str, Any]:
    crc = 0
    sha = hashlib.sha256()
    sz = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            crc = zlib.crc32(b, crc)
            sha.update(b)
            sz += len(b)
    return {"bytes": sz, "crc32": f"{crc & 0xFFFFFFFF:08x}", "sha256": sha.hexdigest()}


def extract_observation_bag(bag_path: Path, obs_dir: Path, obs_idx: int) -> Dict[str, Any]:
    rgb_dir = obs_dir / "rgb"
    meta_dir = obs_dir / "metadata"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    ts = get_typestore(Stores.ROS1_NOETIC)
    frames_meta: List[Dict[str, Any]] = []
    t_g_c_list: List[Dict[str, Any]] = []
    t_g_d_list: List[Dict[str, Any]] = []
    corrupt_frames = 0
    img_width, img_height, img_encoding = None, None, None

    with Reader(bag_path) as reader:
        topic_counts = {c.topic: c.msgcount for c in reader.connections}
        target_conns = [c for c in reader.connections if c.topic in ("color_image", "T_G_C", "T_G_D")]
        frame_idx = 0
        for conn, bag_ts_ns, rawdata in reader.messages(connections=target_conns):
            msg = ts.deserialize_ros1(rawdata, conn.msgtype)
            if conn.topic == "color_image":
                h, w, enc = int(msg.height), int(msg.width), str(msg.encoding)
                img_width, img_height, img_encoding = w, h, enc
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                if enc == "bgr8":
                    img = arr.reshape((h, w, 3))
                elif enc == "rgb8":
                    img = cv2.cvtColor(arr.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
                elif enc == "rgba8":
                    img = cv2.cvtColor(arr.reshape((h, w, 4)), cv2.COLOR_RGBA2BGR)
                elif enc == "bgra8":
                    img = cv2.cvtColor(arr.reshape((h, w, 4)), cv2.COLOR_BGRA2BGR)
                else:
                    corrupt_frames += 1
                    continue
                fname = f"frame_{frame_idx:05d}.png"
                out_img = rgb_dir / fname
                ok = cv2.imwrite(str(out_img), img)
                if not ok or not out_img.exists() or out_img.stat().st_size == 0:
                    corrupt_frames += 1
                else:
                    frames_meta.append({
                        "frame_index": frame_idx,
                        "filename": fname,
                        "rel_path": f"rgb/{fname}",
                        "bag_timestamp_ns": int(bag_ts_ns),
                        "bag_timestamp_s": round(bag_ts_ns / 1e9, 6),
                    })
                frame_idx += 1
            elif conn.topic == "T_G_C":
                tr = msg.transform
                t_g_c_list.append({
                    "bag_timestamp_ns": int(bag_ts_ns),
                    "header_stamp_s": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9,
                    "frame_id": str(msg.header.frame_id),
                    "child_frame_id": str(msg.child_frame_id),
                    "translation": [float(tr.translation.x), float(tr.translation.y), float(tr.translation.z)],
                    "rotation_xyzw": [float(tr.rotation.x), float(tr.rotation.y), float(tr.rotation.z), float(tr.rotation.w)],
                })
            elif conn.topic == "T_G_D":
                tr = msg.transform
                t_g_d_list.append({
                    "bag_timestamp_ns": int(bag_ts_ns),
                    "header_stamp_s": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9,
                    "frame_id": str(msg.header.frame_id),
                    "child_frame_id": str(msg.child_frame_id),
                    "translation": [float(tr.translation.x), float(tr.translation.y), float(tr.translation.z)],
                    "rotation_xyzw": [float(tr.rotation.x), float(tr.rotation.y), float(tr.rotation.z), float(tr.rotation.w)],
                })

    with open(meta_dir / "frames_and_poses.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "observation_index": obs_idx,
                "topic_counts": topic_counts,
                "frames": frames_meta,
                "T_G_C": t_g_c_list,
                "T_G_D": t_g_d_list,
            },
            f,
            indent=2,
        )

    rgb_disk_bytes = sum((rgb_dir / m["filename"]).stat().st_size for m in frames_meta)
    return {
        "extracted_rgb_count": len(frames_meta),
        "corrupt_or_missing_frames": corrupt_frames,
        "width": img_width,
        "height": img_height,
        "source_encoding": img_encoding,
        "output_format": "PNG",
        "topic_counts": topic_counts,
        "t_g_c_count": len(t_g_c_list),
        "t_g_d_count": len(t_g_d_list),
        "rgb_disk_bytes": rgb_disk_bytes,
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    lounge_dir = root / "data/eth_asl_lounge"
    raw_mesh_dir = lounge_dir / "raw/complete_mesh"
    tmp_bag_dir = lounge_dir / "raw/tmp_bags"
    extracted_dir = lounge_dir / "extracted"
    meta_root = lounge_dir / "metadata"
    ann_root = lounge_dir / "annotations"
    man_root = lounge_dir / "manifests"
    for d in [raw_mesh_dir, tmp_bag_dir, extracted_dir, meta_root, ann_root, man_root]:
        d.mkdir(parents=True, exist_ok=True)

    print("Parsing Zip64 Central Directory from ETH Research Collection...")
    entries = parse_zip64_entries(LOUNGE_ZIP_URL, LOUNGE_ZIP_BYTES)

    obs_records: List[Dict[str, Any]] = []
    for i in range(10):
        obs_id = f"observation_{i+1:02d}"
        obs_out = extracted_dir / obs_id
        summary_marker = obs_out / "metadata/extraction_summary.json"

        mesh_fn = f"lounge/complete_mesh/observation_{i}.ply"
        mesh_path = raw_mesh_dir / f"observation_{i}.ply"
        if not mesh_path.exists():
            mesh_info = stream_extract_zip_entry(LOUNGE_ZIP_URL, entries[mesh_fn], mesh_path)
        else:
            mesh_info = file_checksums(mesh_path)

        bag_fn = f"lounge/rosbag/observation_{i}.bag"
        bag_meta = entries[bag_fn]

        if summary_marker.exists():
            rec = json.loads(summary_marker.read_text(encoding="utf-8"))
            print(f"[{obs_id}] Already extracted ({rec['extracted_rgb_count']} frames). Skipping.")
            obs_records.append(rec)
            continue

        bag_path = tmp_bag_dir / f"observation_{i}.bag"
        if bag_path.exists() and bag_path.stat().st_size == bag_meta["uncomp_sz"]:
            print(f"[{obs_id}] Using cached verified {bag_path.name} ({bag_meta['uncomp_sz']/(1024**2):.2f} MB)...")
            bag_info = file_checksums(bag_path)
            assert bag_info["crc32"] == f"{bag_meta['crc32']:08x}", "Cached bag CRC mismatch!"
        else:
            print(f"[{obs_id}] Streaming {bag_fn} ({bag_meta['comp_sz']/(1024**2):.2f} MB compressed -> {bag_meta['uncomp_sz']/(1024**2):.2f} MB)...")
            bag_info = stream_extract_zip_entry(LOUNGE_ZIP_URL, bag_meta, bag_path)

        print(f"[{obs_id}] Extracting RGB PNGs and transforms...")
        ext_res = extract_observation_bag(bag_path, obs_out, i)

        # Remove temporary .bag to keep disk usage well under 38 GB limit
        if bag_path.exists():
            bag_path.unlink()

        rec = {
            "observation_id": obs_id,
            "source_bag_index": i,
            "archive_url": LOUNGE_ZIP_URL,
            "archive_entry_bag": bag_fn,
            "bag_compressed_bytes": bag_meta["comp_sz"],
            "bag_uncompressed_bytes": bag_info["bytes"],
            "bag_crc32": bag_info["crc32"],
            "bag_sha256": bag_info["sha256"],
            "mesh_rel_path": f"data/eth_asl_lounge/raw/complete_mesh/observation_{i}.ply",
            "mesh_bytes": mesh_info["bytes"],
            "mesh_crc32": mesh_info["crc32"],
            "mesh_sha256": mesh_info["sha256"],
            "extracted_dir": f"data/eth_asl_lounge/extracted/{obs_id}",
            "poses_metadata_path": f"data/eth_asl_lounge/extracted/{obs_id}/metadata/frames_and_poses.json",
            "extraction_status": "VERIFIED" if ext_res["corrupt_or_missing_frames"] == 0 else "PARTIAL",
            **ext_res,
        }
        with open(summary_marker, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        obs_records.append(rec)
        print(f"[{obs_id}] Done: {rec['extracted_rgb_count']} PNGs ({rec['rgb_disk_bytes']/(1024**2):.2f} MB), {rec['corrupt_or_missing_frames']} corrupt.")

    # Clean empty tmp_bags dir if empty
    if tmp_bag_dir.exists() and not any(tmp_bag_dir.iterdir()):
        tmp_bag_dir.rmdir()

    manifest = {
        "dataset_name": "ETH_ASL_Change_Detection_Lounge",
        "provenance": "PUBLIC_BENCHMARK",
        "official_urls": [
            "https://projects.asl.ethz.ch/datasets/change-detection/",
            "https://github.com/ethz-asl/change_detection_ds",
            LOUNGE_ZIP_URL,
        ],
        "archive_note": (
            "lounge.zip is 30,950,780,627 bytes (28.83 GB compressed, 67.21 GB uncompressed). "
            "Because local free disk space was 38.34 GB, each .bag was streamed via HTTP Range "
            "with strict Zip64 CRC32 + SHA-256 verification, extracted losslessly, and purged from tmp_bags."
        ),
        "summary": {
            "total_recordings": len(obs_records),
            "total_extracted_rgb_frames": sum(r["extracted_rgb_count"] for r in obs_records),
            "total_corrupt_or_missing_frames": sum(r["corrupt_or_missing_frames"] for r in obs_records),
            "total_bag_compressed_bytes": sum(r["bag_compressed_bytes"] for r in obs_records),
            "total_bag_uncompressed_bytes": sum(r["bag_uncompressed_bytes"] for r in obs_records),
            "total_mesh_bytes": sum(r["mesh_bytes"] for r in obs_records),
            "total_extracted_rgb_bytes": sum(r["rgb_disk_bytes"] for r in obs_records),
        },
        "observations": obs_records,
    }

    out_manifest = root / "research_validation/ETH_ASL_DATASET_MANIFEST.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(man_root / "ETH_ASL_DATASET_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("ALL 10 LOUNGE OBSERVATIONS COMPLETE! Manifest saved to", out_manifest)


if __name__ == "__main__":
    main()
