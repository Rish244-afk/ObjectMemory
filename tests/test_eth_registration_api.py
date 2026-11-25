"""
Tests for the supplementary ETH ASL Lounge registration endpoints.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from objectmemory.api.eth_registration import create_eth_router


def _make_textured_image(seed: int = 42, shift_x: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(120):
        x = int(rng.integers(20, 280)) + shift_x
        y = int(rng.integers(20, 210))
        r = int(rng.integers(5, 18))
        col = tuple(int(c) for c in rng.integers(40, 255, size=3))
        cv2.circle(img, (max(0, min(319, x)), y), r, col, -1)
        cv2.rectangle(img, (max(0, min(300, x)), y), (max(0, min(319, x + 12)), min(239, y + 12)), (255, 255, 255), 1)
    return img


class TestEthRegistrationAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.eth_root = Path(self.tmp_dir.name)
        rgb_dir = self.eth_root / "extracted" / "observation_01" / "rgb"
        meta_dir = self.eth_root / "extracted" / "observation_01" / "metadata"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "frames_and_poses.json").write_text("{}", encoding="utf-8")

        # Save identical/textured pair (should succeed) and blank frame (should fail safely)
        img0 = _make_textured_image(seed=10, shift_x=0)
        img1 = _make_textured_image(seed=10, shift_x=0)
        blank = np.zeros((240, 320, 3), dtype=np.uint8)

        cv2.imwrite(str(rgb_dir / "frame_00000.png"), img0)
        cv2.imwrite(str(rgb_dir / "frame_00001.png"), img1)
        cv2.imwrite(str(rgb_dir / "frame_00002.png"), blank)

        app = FastAPI()
        app.include_router(create_eth_router(dataset_root=self.eth_root))
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_session_and_frame_discovery(self) -> None:
        r = self.client.get("/api/eth-lounge/sessions")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["available"])
        self.assertEqual(data["evaluation_role"], "SUPPLEMENTARY_REGISTRATION_ONLY")
        self.assertEqual(len(data["sessions"]), 1)
        self.assertEqual(data["sessions"][0]["session_id"], "observation_01")
        self.assertEqual(data["sessions"][0]["frame_count"], 3)

        rf = self.client.get("/api/eth-lounge/sessions/observation_01/frames")
        self.assertEqual(rf.status_code, 200)
        fdata = rf.json()
        self.assertEqual(fdata["total_frames"], 3)
        self.assertEqual([f["filename"] for f in fdata["frames"]], ["frame_00000.png", "frame_00001.png", "frame_00002.png"])

        ri = self.client.get("/api/eth-lounge/sessions/observation_01/frames/frame_00000.png")
        self.assertEqual(ri.status_code, 200)
        self.assertEqual(ri.headers["content-type"], "image/png")

    def test_invalid_inputs_rejected(self) -> None:
        r1 = self.client.get("/api/eth-lounge/sessions/bad_session/frames")
        self.assertEqual(r1.status_code, 400)

        r2 = self.client.get("/api/eth-lounge/sessions/observation_01/frames/..%2Fsecret.png")
        self.assertIn(r2.status_code, (400, 404))

        r3 = self.client.get("/api/eth-lounge/sessions/observation_01/frames/bad_frame.txt")
        self.assertEqual(r3.status_code, 400)

    def test_registration_success_and_failure_handling(self) -> None:
        # 1. Success on textured pair
        r_ok = self.client.post(
            "/api/eth-lounge/register",
            json={
                "ref_session_id": "observation_01",
                "ref_frame": "frame_00000.png",
                "query_session_id": "observation_01",
                "query_frame": "frame_00001.png",
            },
        )
        self.assertEqual(r_ok.status_code, 200)
        d_ok = r_ok.json()
        self.assertTrue(d_ok["success"])
        self.assertEqual(d_ok["status"], "SUCCESS")
        self.assertGreater(d_ok["num_inliers"], 0)
        self.assertIsNotNone(d_ok["aligned_image_data_url"])
        self.assertIsNone(d_ok["unreliable_warning"])

        # 2. Safe failure on featureless query frame
        r_fail = self.client.post(
            "/api/eth-lounge/register",
            json={
                "ref_session_id": "observation_01",
                "ref_frame": "frame_00000.png",
                "query_session_id": "observation_01",
                "query_frame": "frame_00002.png",
            },
        )
        self.assertEqual(r_fail.status_code, 200)
        d_fail = r_fail.json()
        self.assertFalse(d_fail["success"])
        self.assertIsNotNone(d_fail["unreliable_warning"])
        self.assertIsNone(d_fail["aligned_image_data_url"])

    def test_real_eth_dataset_smoke_if_present(self) -> None:
        real_root = Path("data/eth_asl_lounge")
        if not (real_root / "extracted" / "observation_01" / "rgb" / "frame_00000.png").exists():
            self.skipTest("ETH ASL Lounge dataset not extracted locally.")

        app = FastAPI()
        app.include_router(create_eth_router(dataset_root=real_root))
        client = TestClient(app)

        rs = client.get("/api/eth-lounge/sessions").json()
        self.assertTrue(rs["available"])
        self.assertEqual(len(rs["sessions"]), 10)

        # Adjacent pair: observation_01 frame_00000 -> frame_00001
        r_adj = client.post(
            "/api/eth-lounge/register",
            json={
                "ref_session_id": "observation_01",
                "ref_frame": "frame_00000.png",
                "query_session_id": "observation_01",
                "query_frame": "frame_00001.png",
            },
        ).json()
        self.assertTrue(r_adj["success"])
        self.assertEqual(r_adj["status"], "SUCCESS")

        # 150-frame gap: observation_01 frame_00000 -> frame_00150 (known failure)
        r_gap = client.post(
            "/api/eth-lounge/register",
            json={
                "ref_session_id": "observation_01",
                "ref_frame": "frame_00000.png",
                "query_session_id": "observation_01",
                "query_frame": "frame_00150.png",
            },
        ).json()
        self.assertFalse(r_gap["success"])
        self.assertIsNotNone(r_gap["unreliable_warning"])
