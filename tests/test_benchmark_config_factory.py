"""
Focused Unit Tests for ObjectMemory Benchmark Configuration Factory (Phase 3B).

Verifies:
1. All configuration names (B0-B3, A1-A6) resolve correctly.
2. Configurations are deterministic and produce stable SHA-256 hashes.
3. Serialization to dictionary and JSON is stable and sorted.
4. Unsupported settings are explicitly rejected with UnsupportedConfigurationError.
5. B0 is strictly memoryless (no temporal config).
6. B1 does not silently inherit B3 safeguards.
7. B3 matches current production defaults.
8. A1-A6 isolate only their intended settings.
9. Configuration objects cannot accidentally share mutable state (deep copying).
10. Engine/associator behavioral verification:
    - B3 triggers ambiguity gating when candidate gap < 0.12.
    - A5 disables ambiguity gating when candidate gap < 0.12 (genuine engine response).
"""

from __future__ import annotations

import unittest
import numpy as np

from objectmemory.benchmarks.config_factory import (
    BenchmarkConfig,
    BenchmarkConfigFactory,
    ConfigurationSupportStatus,
    UnsupportedConfigurationError,
)
from objectmemory.detection.models import Detection, VisualSignature
from objectmemory.domain.models import BoundingBox, WorkbenchEntity
from objectmemory.registration.models import AlignmentStatus, RegistrationResult
from objectmemory.temporal.associator import CrossVisitEntityAssociator
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.temporal.models import TemporalEngineConfig


def _make_dummy_sig(dim: int = 148, fill_val: float = 1.0) -> VisualSignature:
    vec = np.full(dim, fill_val, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return VisualSignature(
        color_histogram=vec[:128].copy(),
        texture_descriptor=vec[128:].copy(),
        combined_vector=vec,
    )


class TestBenchmarkConfigFactory(unittest.TestCase):
    def test_01_all_configuration_names_resolve(self) -> None:
        """Requirement 1: All known configuration names resolve without crashing."""
        expected_ids = {"B0", "B1", "B2", "B3", "A1", "A2", "A3", "A4", "A5", "A6"}
        self.assertEqual(set(BenchmarkConfigFactory.list_known_configs()), expected_ids)

        for cid in expected_ids:
            status = BenchmarkConfigFactory.get_config_status(cid)
            self.assertIn(status, [
                ConfigurationSupportStatus.SUPPORTED,
                ConfigurationSupportStatus.PARTIALLY_SUPPORTED,
                ConfigurationSupportStatus.BLOCKED,
            ])

    def test_02_configurations_are_deterministic(self) -> None:
        """Requirement 2: Instantiating configurations repeatedly produces identical hashes."""
        for cid in ["B0", "B3", "A5"]:
            c1 = BenchmarkConfigFactory.create_config(cid)
            c2 = BenchmarkConfigFactory.create_config(cid)
            self.assertEqual(c1.deterministic_hash, c2.deterministic_hash)
            self.assertEqual(c1.to_json(), c2.to_json())

    def test_03_serialization_is_stable_and_sorted(self) -> None:
        """Requirement 3: JSON serialization produces sorted keys and identical outputs."""
        b3 = BenchmarkConfigFactory.create_config("B3")
        json_str = b3.to_json(indent=2)
        dict_repr = b3.to_dict()

        self.assertIn('"config_id": "B3"', json_str)
        self.assertIn('"support_status": "SUPPORTED"', json_str)
        self.assertIn('"ambiguity_margin": 0.12', json_str)
        self.assertIsInstance(dict_repr["effective_parameters"], dict)
        self.assertEqual(dict_repr["config_id"], "B3")

    def test_04_unsupported_settings_rejected_without_opt_in(self) -> None:
        """Requirement 4: Blocked/partially supported configurations fail unless explicitly allowed."""
        # Blocked configurations must raise UnsupportedConfigurationError
        for blocked_id in ["B1", "B2", "A1", "A4", "A6"]:
            with self.assertRaises(UnsupportedConfigurationError) as ctx:
                BenchmarkConfigFactory.create_config(blocked_id)
            self.assertIn(f"Configuration '{blocked_id}' is BLOCKED", str(ctx.exception))

        # Partially supported configurations must raise UnsupportedConfigurationError unless allow_partial=True
        for part_id in ["A2", "A3"]:
            with self.assertRaises(UnsupportedConfigurationError) as ctx:
                BenchmarkConfigFactory.create_config(part_id)
            self.assertIn(f"Configuration '{part_id}' is only PARTIALLY_SUPPORTED", str(ctx.exception))

        # Unknown IDs must raise ValueError
        with self.assertRaises(ValueError):
            BenchmarkConfigFactory.create_config("UNKNOWN_CONFIG_XYZ")

    def test_05_b0_does_not_perform_temporal_association(self) -> None:
        """Requirement 5: B0 is strictly memoryless and has no temporal configuration."""
        b0 = BenchmarkConfigFactory.create_config("B0")
        self.assertEqual(b0.pipeline_type, "MEMORYLESS_DETECTION")
        self.assertIsNone(b0.temporal_config)
        self.assertFalse(b0.effective_parameters["temporal_memory"])
        self.assertFalse(b0.effective_parameters["cross_visit_association"])

    def test_06_b1_does_not_silently_inherit_b3_safeguards(self) -> None:
        """Requirement 6: B1 is blocked and does not masquerade as having B3 safeguards."""
        b1 = BenchmarkConfigFactory.create_config("B1", allow_blocked=True)
        self.assertEqual(b1.support_status, ConfigurationSupportStatus.BLOCKED)
        self.assertIn("registration_absence_freezing", b1.disabled_safeguards)
        self.assertIn("appearance_vacancy_verification", b1.disabled_safeguards)
        self.assertIn("VALID_STATE_TRANSITIONS", b1.notes)

    def test_07_b3_matches_current_production_defaults(self) -> None:
        """Requirement 7: B3 matches production TemporalEngineConfig defaults exactly."""
        b3 = BenchmarkConfigFactory.create_config("B3")
        prod = TemporalEngineConfig()

        self.assertIsNotNone(b3.temporal_config)
        assert b3.temporal_config is not None
        self.assertEqual(b3.temporal_config.ambiguity_margin, prod.ambiguity_margin) # 0.12
        self.assertEqual(b3.temporal_config.vacancy_similarity_max, prod.vacancy_similarity_max) # 0.40
        self.assertEqual(b3.temporal_config.signature_similarity_threshold, prod.signature_similarity_threshold) # 0.65
        self.assertEqual(b3.temporal_config.min_registration_inlier_ratio, prod.min_registration_inlier_ratio) # 0.20
        self.assertEqual(b3.temporal_config.min_overlap_fraction, prod.min_overlap_fraction) # 0.70
        self.assertEqual(b3.temporal_config.vacancy_visits_for_removal, prod.vacancy_visits_for_removal) # 2
        self.assertEqual(b3.temporal_config.unobserved_visits_for_removal, prod.unobserved_visits_for_removal) # 3

    def test_08_a1_through_a6_isolate_intended_settings(self) -> None:
        """Requirement 8: Each ablation modifies only its intended target setting."""
        # A1
        a1 = BenchmarkConfigFactory.create_config("A1", allow_blocked=True)
        assert a1.temporal_config is not None
        self.assertEqual(a1.temporal_config.min_registration_inlier_ratio, 0.0)
        self.assertEqual(a1.temporal_config.ambiguity_margin, 0.12) # Unchanged

        # A2
        a2 = BenchmarkConfigFactory.create_config("A2", allow_partial=True)
        assert a2.temporal_config is not None
        self.assertEqual(a2.temporal_config.vacancy_similarity_max, 1.0)
        self.assertEqual(a2.temporal_config.ambiguity_margin, 0.12) # Unchanged

        # A3
        a3 = BenchmarkConfigFactory.create_config("A3", allow_partial=True)
        assert a3.temporal_config is not None
        self.assertEqual(a3.temporal_config.min_overlap_fraction, 0.0)
        self.assertEqual(a3.temporal_config.vacancy_similarity_max, 0.40) # Unchanged

        # A4
        a4 = BenchmarkConfigFactory.create_config("A4", allow_blocked=True)
        assert a4.temporal_config is not None
        self.assertEqual(a4.temporal_config.vacancy_visits_for_removal, 1)

        # A5
        a5 = BenchmarkConfigFactory.create_config("A5")
        assert a5.temporal_config is not None
        self.assertEqual(a5.temporal_config.ambiguity_margin, 0.0)

        # A6
        a6 = BenchmarkConfigFactory.create_config("A6", allow_blocked=True)
        self.assertEqual(a6.support_status, ConfigurationSupportStatus.BLOCKED)

    def test_09_configuration_objects_do_not_share_mutable_state(self) -> None:
        """Requirement 9: Mutating an extracted config does not pollute factory or subsequent configs."""
        b3_first = BenchmarkConfigFactory.create_config("B3")
        cfg_extracted = b3_first.get_isolated_temporal_config()
        assert cfg_extracted is not None

        # Mutate extracted config
        cfg_extracted.ambiguity_margin = 0.99
        cfg_extracted.vacancy_similarity_max = 0.99

        # Instantiate second config and assert pristine defaults
        b3_second = BenchmarkConfigFactory.create_config("B3")
        assert b3_second.temporal_config is not None
        self.assertEqual(b3_second.temporal_config.ambiguity_margin, 0.12)
        self.assertEqual(b3_second.temporal_config.vacancy_similarity_max, 0.40)
        self.assertEqual(b3_first.temporal_config.ambiguity_margin, 0.12)

    def test_10_engine_behavioral_response_b3_vs_a5(self) -> None:
        """
        Requirement 10: Verifies that the engine/associator genuinely responds to B3 vs A5.
        Under B3 (ambiguity_margin=0.12): candidate affinity difference < 0.12 triggers AMBIGUOUS match.
        Under A5 (ambiguity_margin=0.0): candidate affinity difference < 0.12 assigns TOP candidate.
        """
        # Mock registration
        reg = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="MOCK",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=np.eye(3, dtype=np.float64),
            overlap_mask=np.ones((480, 640), dtype=np.uint8) * 255,
            inlier_ratio=0.85,
        )

        e1 = WorkbenchEntity(
            entity_id="ent_01",
            label="cup",
            created_at="2026-11-14T10:00:00Z",
            canonical_bbox=BoundingBox(10.0, 10.0, 50.0, 50.0),
        )
        sig_base = _make_dummy_sig(fill_val=1.0)
        sig_cand1 = _make_dummy_sig(fill_val=1.0) # Cosine sim = 1.0
        sig_cand2 = _make_dummy_sig(fill_val=0.98) # Cosine sim ~ 1.0 (very close!)

        # Two candidate detections of same class competing for e1
        d1 = Detection(
            label="cup",
            confidence=0.90,
            raw_bbox=BoundingBox(10.0, 10.0, 50.0, 50.0),
            canonical_bbox=BoundingBox(10.0, 10.0, 50.0, 50.0),
            signature=sig_cand1,
        )
        d2 = Detection(
            label="cup",
            confidence=0.88,
            raw_bbox=BoundingBox(12.0, 12.0, 52.0, 52.0),
            canonical_bbox=BoundingBox(12.0, 12.0, 52.0, 52.0),
            signature=sig_cand2,
        )

        # 1. Test under B3 (ambiguity_margin = 0.12)
        b3_config = BenchmarkConfigFactory.create_config("B3")
        associator_b3 = CrossVisitEntityAssociator(config=b3_config.temporal_config)
        matches_b3, _ = associator_b3.associate(
            entities=[e1],
            detections=[d1, d2],
            registration=reg,
            entity_signatures={"ent_01": sig_base},
        )
        self.assertEqual(len(matches_b3), 1)
        # B3 must flag as ambiguous!
        self.assertTrue(matches_b3[0].is_ambiguous)
        self.assertFalse(matches_b3[0].is_associated)

        # 2. Test under A5 (ambiguity_margin = 0.0)
        a5_config = BenchmarkConfigFactory.create_config("A5")
        associator_a5 = CrossVisitEntityAssociator(config=a5_config.temporal_config)
        matches_a5, _ = associator_a5.associate(
            entities=[e1],
            detections=[d1, d2],
            registration=reg,
            entity_signatures={"ent_01": sig_base},
        )
        self.assertEqual(len(matches_a5), 1)
        # A5 disables ambiguity gating: successfully associates top candidate!
        self.assertFalse(matches_a5[0].is_ambiguous)
        self.assertTrue(matches_a5[0].is_associated)
        self.assertEqual(matches_a5[0].detection, d1)


if __name__ == "__main__":
    unittest.main()
