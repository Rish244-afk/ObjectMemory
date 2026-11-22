"""
Temporal benchmark sequence dataset generator for ObjectMemory Phase 5.

Produces controlled multi-visit sequences (>= 3 visits each) representing 10 environmental conditions:
1. Unchanged scene across multiple visits (unchanged_multi_visit).
2. Object moved and remaining at new location (moved_multi_visit).
3. Object genuinely removed and remaining absent across repeated visits (removed_multi_visit).
4. Temporarily missed detection followed by successful redetection (missed_detection_redetection).
5. Partial occlusion followed by visibility recovery (occlusion_recovery).
6. Lighting changes without physical object movement (lighting_shifts_multi_visit).
7. Ambiguous, visually similar objects (ambiguous_objects_multi_visit).
8. Invalid or failed registration (registration_failure_multi_visit).
9. Object becoming visible after being outside the camera frame (out_of_frame_reentry).
10. Removed object later returning to the workbench (removed_object_returning).

Safeguard:
Every sequence includes explicit ground truth, sequence metadata, and is_synthetic = True flag.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from objectmemory.benchmarks.synthetic_dataset import (
    _draw_component_box,
    _draw_multimeter,
    _draw_pcb,
    _draw_resistor,
    _draw_wire_strippers,
    _draw_workbench_surface,
)
from objectmemory.domain.models import BoundingBox
from objectmemory.domain.states import EpistemicState


@dataclass
class TemporalGroundTruthEntity:
    """
    Ground truth state trajectory for one physical object across visits.
    """
    entity_label: str
    expected_states: List[EpistemicState]  # Expected state at visit t: 0, 1, 2...
    initial_bbox: BoundingBox
    final_bbox: Optional[BoundingBox] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_label": self.entity_label,
            "expected_states": [s.value for s in self.expected_states],
            "initial_bbox": [self.initial_bbox.x_min, self.initial_bbox.y_min, self.initial_bbox.x_max, self.initial_bbox.y_max],
            "final_bbox": [self.final_bbox.x_min, self.final_bbox.y_min, self.final_bbox.x_max, self.final_bbox.y_max] if self.final_bbox else None,
            "notes": self.notes,
        }


@dataclass
class BenchmarkVisitFrame:
    """
    Single visit frame within a multi-visit sequence.
    """
    visit_index: int
    session_id: str
    timestamp: str
    image: np.ndarray
    expected_alignment_success: bool
    ground_truth_bboxes: Dict[str, Optional[BoundingBox]]  # entity_label -> bbox or None
    is_occluded: Dict[str, bool] = field(default_factory=dict)
    is_missed_detection: Dict[str, bool] = field(default_factory=dict)


@dataclass
class SyntheticTemporalSequence:
    """
    Full multi-visit sequence representing one controlled experimental benchmark case.
    """
    sequence_name: str
    description: str
    visits: List[BenchmarkVisitFrame]
    ground_truth_entities: List[TemporalGroundTruthEntity]
    random_seed: int
    is_synthetic: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class SyntheticTemporalSequenceGenerator:
    """
    Generates reproducible multi-visit sequences for all 10 temporal scenarios.
    """

    SUPPORTED_SCENARIOS = [
        "unchanged_multi_visit",
        "moved_multi_visit",
        "removed_multi_visit",
        "missed_detection_redetection",
        "occlusion_recovery",
        "lighting_shifts_multi_visit",
        "ambiguous_objects_multi_visit",
        "registration_failure_multi_visit",
        "out_of_frame_reentry",
        "removed_object_returning",
    ]

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height

    def generate_all_sequences(self, base_seed: int = 42) -> List[SyntheticTemporalSequence]:
        """Generates all 10 multi-visit sequences deterministically."""
        return [
            self.generate_sequence(name, seed=base_seed + idx * 23)
            for idx, name in enumerate(self.SUPPORTED_SCENARIOS)
        ]

    def generate_sequence(self, name: str, seed: int = 42) -> SyntheticTemporalSequence:
        """Generates a single controlled sequence by name."""
        if name == "unchanged_multi_visit":
            return self._gen_unchanged_multi_visit(seed)
        elif name == "moved_multi_visit":
            return self._gen_moved_multi_visit(seed)
        elif name == "removed_multi_visit":
            return self._gen_removed_multi_visit(seed)
        elif name == "missed_detection_redetection":
            return self._gen_missed_detection_redetection(seed)
        elif name == "occlusion_recovery":
            return self._gen_occlusion_recovery(seed)
        elif name == "lighting_shifts_multi_visit":
            return self._gen_lighting_shifts_multi_visit(seed)
        elif name == "ambiguous_objects_multi_visit":
            return self._gen_ambiguous_objects_multi_visit(seed)
        elif name == "registration_failure_multi_visit":
            return self._gen_registration_failure_multi_visit(seed)
        elif name == "out_of_frame_reentry":
            return self._gen_out_of_frame_reentry(seed)
        elif name == "removed_object_returning":
            return self._gen_removed_object_returning(seed)
        else:
            raise ValueError(f"Unknown sequence '{name}'. Supported: {self.SUPPORTED_SCENARIOS}")

    # -------------------------------------------------------------------------
    # Scenario Implementations (3 or 4 visits each)
    # -------------------------------------------------------------------------

    def _gen_unchanged_multi_visit(self, seed: int) -> SyntheticTemporalSequence:
        """Sequence 1: Unchanged scene across 3 visits."""
        m_box = (280, 80, 390, 260)
        p_box = (60, 60, 220, 190)

        visits = []
        for v_idx, ts in enumerate(["2026-03-01T10:00:00Z", "2026-03-02T10:00:00Z", "2026-03-03T10:00:00Z"]):
            img = _draw_workbench_surface(self.width, self.height, seed=seed)
            _draw_multimeter(img, *m_box)
            _draw_pcb(img, *p_box)
            visits.append(
                BenchmarkVisitFrame(
                    visit_index=v_idx,
                    session_id=f"unchanged_v{v_idx + 1}",
                    timestamp=ts,
                    image=img,
                    expected_alignment_success=True,
                    ground_truth_bboxes={
                        "multimeter": BoundingBox(*m_box),
                        "pcb": BoundingBox(*p_box),
                    },
                )
            )

        entities = [
            TemporalGroundTruthEntity(
                "multimeter",
                [EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT],
                BoundingBox(*m_box),
                BoundingBox(*m_box),
            ),
            TemporalGroundTruthEntity(
                "pcb",
                [EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT],
                BoundingBox(*p_box),
                BoundingBox(*p_box),
            ),
        ]
        return SyntheticTemporalSequence(
            "unchanged_multi_visit",
            "Objects remain unchanged across 3 consecutive daily visits.",
            visits,
            entities,
            seed,
        )

    def _gen_moved_multi_visit(self, seed: int) -> SyntheticTemporalSequence:
        """Sequence 2: Object moved in visit 2 and remaining at new location in visit 3."""
        m_box1 = (280, 80, 390, 260)
        m_box2 = (440, 200, 550, 380)  # Relocated by > 180px
        p_box = (60, 60, 220, 190)

        visits = []
        # Visit 1: Initial position
        img1 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img1, *m_box1)
        _draw_pcb(img1, *p_box)
        visits.append(BenchmarkVisitFrame(0, "moved_v1", "2026-03-01T10:00:00Z", img1, True, {"multimeter": BoundingBox(*m_box1), "pcb": BoundingBox(*p_box)}))

        # Visit 2: Moved
        img2 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img2, *m_box2)
        _draw_pcb(img2, *p_box)
        visits.append(BenchmarkVisitFrame(1, "moved_v2", "2026-03-02T10:00:00Z", img2, True, {"multimeter": BoundingBox(*m_box2), "pcb": BoundingBox(*p_box)}))

        # Visit 3: Remains at new position
        img3 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img3, *m_box2)
        _draw_pcb(img3, *p_box)
        visits.append(BenchmarkVisitFrame(2, "moved_v3", "2026-03-03T10:00:00Z", img3, True, {"multimeter": BoundingBox(*m_box2), "pcb": BoundingBox(*p_box)}))

        entities = [
            TemporalGroundTruthEntity(
                "multimeter",
                [EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_MOVED, EpistemicState.CONFIRMED_MOVED],
                BoundingBox(*m_box1),
                BoundingBox(*m_box2),
            ),
            TemporalGroundTruthEntity(
                "pcb",
                [EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT],
                BoundingBox(*p_box),
                BoundingBox(*p_box),
            ),
        ]
        return SyntheticTemporalSequence(
            "moved_multi_visit",
            "Multimeter is relocated in visit 2 and remains at the new location in visit 3.",
            visits,
            entities,
            seed,
        )

    def _gen_removed_multi_visit(self, seed: int) -> SyntheticTemporalSequence:
        """
        Sequence 3: Object genuinely removed and remaining absent across repeated reliable visits.
        Progression: CONFIRMED_PRESENT -> PENDING_REMOVAL -> CONFIRMED_REMOVED.
        """
        m_box = (280, 80, 390, 260)
        p_box = (60, 60, 220, 190)

        visits = []
        # Visit 1: Present
        img1 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img1, *m_box)
        _draw_pcb(img1, *p_box)
        visits.append(BenchmarkVisitFrame(0, "removed_v1", "2026-03-01T10:00:00Z", img1, True, {"multimeter": BoundingBox(*m_box), "pcb": BoundingBox(*p_box)}))

        # Visit 2: Removed (streak = 1, PENDING_REMOVAL)
        img2 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_pcb(img2, *p_box)
        visits.append(BenchmarkVisitFrame(1, "removed_v2", "2026-03-02T10:00:00Z", img2, True, {"multimeter": None, "pcb": BoundingBox(*p_box)}))

        # Visit 3: Still removed (streak = 2, CONFIRMED_REMOVED)
        img3 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_pcb(img3, *p_box)
        visits.append(BenchmarkVisitFrame(2, "removed_v3", "2026-03-03T10:00:00Z", img3, True, {"multimeter": None, "pcb": BoundingBox(*p_box)}))

        entities = [
            TemporalGroundTruthEntity(
                "multimeter",
                [EpistemicState.CONFIRMED_PRESENT, EpistemicState.PENDING_REMOVAL, EpistemicState.CONFIRMED_REMOVED],
                BoundingBox(*m_box),
                None,
            ),
            TemporalGroundTruthEntity(
                "pcb",
                [EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT],
                BoundingBox(*p_box),
                BoundingBox(*p_box),
            ),
        ]
        return SyntheticTemporalSequence(
            "removed_multi_visit",
            "Multimeter is removed in visit 2 and verified absent across visits 2 and 3.",
            visits,
            entities,
            seed,
        )

    def _gen_missed_detection_redetection(self, seed: int) -> SyntheticTemporalSequence:
        """
        Sequence 4: A temporarily missed detection followed by successful redetection.
        Crucial safeguard: A single missed detection must not confirm removal!
        """
        m_box = (280, 80, 390, 260)
        p_box = (60, 60, 220, 190)

        visits = []
        # Visit 1: Detected
        img1 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img1, *m_box)
        _draw_pcb(img1, *p_box)
        visits.append(BenchmarkVisitFrame(0, "missed_v1", "2026-03-01T10:00:00Z", img1, True, {"multimeter": BoundingBox(*m_box), "pcb": BoundingBox(*p_box)}))

        # Visit 2: Object is physically present on image, but detector misses it!
        # Because object is visible on the image patch, vacancy check sees object-like pixels and does not confirm vacancy!
        img2 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img2, *m_box)
        _draw_pcb(img2, *p_box)
        visits.append(
            BenchmarkVisitFrame(
                1,
                "missed_v2",
                "2026-03-02T10:00:00Z",
                img2,
                True,
                {"multimeter": BoundingBox(*m_box), "pcb": BoundingBox(*p_box)},
                is_missed_detection={"multimeter": True},
            )
        )

        # Visit 3: Detector redetects object -> Remains CONFIRMED_PRESENT
        img3 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img3, *m_box)
        _draw_pcb(img3, *p_box)
        visits.append(BenchmarkVisitFrame(2, "missed_v3", "2026-03-03T10:00:00Z", img3, True, {"multimeter": BoundingBox(*m_box), "pcb": BoundingBox(*p_box)}))

        entities = [
            TemporalGroundTruthEntity(
                "multimeter",
                [EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT],
                BoundingBox(*m_box),
                BoundingBox(*m_box),
                notes="Missed in visit 2; non-detection with patch appearance prevents removal; redetected in visit 3.",
            ),
        ]
        return SyntheticTemporalSequence(
            "missed_detection_redetection",
            "Detector misses multimeter in visit 2, but non-detection is not removal; redetected in visit 3.",
            visits,
            entities,
            seed,
        )

    def _gen_occlusion_recovery(self, seed: int) -> SyntheticTemporalSequence:
        """
        Sequence 5: Partial occlusion followed by visibility recovery.
        """
        m_box = (280, 80, 390, 260)
        c_box = (140, 280, 270, 390)
        # Notebook occluding multimeter
        occ_box = (240, 70, 370, 230)

        visits = []
        # Visit 1: Clear
        img1 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img1, *m_box)
        _draw_component_box(img1, *c_box)
        visits.append(BenchmarkVisitFrame(0, "occl_v1", "2026-03-01T10:00:00Z", img1, True, {"multimeter": BoundingBox(*m_box), "component_box": BoundingBox(*c_box)}))

        # Visit 2: Partially occluded by clipboard/paper
        img2 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img2, *m_box)
        cv2.rectangle(img2, (occ_box[0], occ_box[1]), (occ_box[2], occ_box[3]), (220, 230, 230), -1)
        _draw_component_box(img2, *c_box)
        visits.append(
            BenchmarkVisitFrame(
                1,
                "occl_v2",
                "2026-03-02T10:00:00Z",
                img2,
                True,
                {"multimeter": BoundingBox(*m_box), "component_box": BoundingBox(*c_box)},
                is_occluded={"multimeter": True},
            )
        )

        # Visit 3: Obstruction removed, full visibility recovered
        img3 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img3, *m_box)
        _draw_component_box(img3, *c_box)
        visits.append(BenchmarkVisitFrame(2, "occl_v3", "2026-03-03T10:00:00Z", img3, True, {"multimeter": BoundingBox(*m_box), "component_box": BoundingBox(*c_box)}))

        entities = [
            TemporalGroundTruthEntity(
                "multimeter",
                [EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT],
                BoundingBox(*m_box),
                BoundingBox(*m_box),
                notes="Occluded in visit 2; occlusion prevents removal; fully visible in visit 3.",
            ),
        ]
        return SyntheticTemporalSequence(
            "occlusion_recovery",
            "Multimeter is partially occluded in visit 2 and fully restored in visit 3.",
            visits,
            entities,
            seed,
        )

    def _gen_lighting_shifts_multi_visit(self, seed: int) -> SyntheticTemporalSequence:
        """
        Sequence 6: Lighting shifts without physical object movement.
        """
        m_box = (280, 80, 390, 260)
        p_box = (60, 60, 220, 190)

        visits = []
        # Visit 1: Standard lighting
        img1 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img1, *m_box)
        _draw_pcb(img1, *p_box)
        visits.append(BenchmarkVisitFrame(0, "light_v1", "2026-03-01T10:00:00Z", img1, True, {"multimeter": BoundingBox(*m_box), "pcb": BoundingBox(*p_box)}))

        # Visit 2: Dim afternoon lighting (40% brightness drop)
        img2 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img2, *m_box)
        _draw_pcb(img2, *p_box)
        img2 = np.clip(img2.astype(np.float32) * 0.60, 0, 255).astype(np.uint8)
        visits.append(BenchmarkVisitFrame(1, "light_v2", "2026-03-02T10:00:00Z", img2, True, {"multimeter": BoundingBox(*m_box), "pcb": BoundingBox(*p_box)}))

        # Visit 3: Warm overhead desk lamp lighting
        img3 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img3, *m_box)
        _draw_pcb(img3, *p_box)
        img3_float = img3.astype(np.float32)
        img3_float[:, :, 0] *= 0.80  # Dim blue
        img3_float[:, :, 2] = np.clip(img3_float[:, :, 2] * 1.25, 0, 255)  # Boost red
        img3 = np.clip(img3_float, 0, 255).astype(np.uint8)
        visits.append(BenchmarkVisitFrame(2, "light_v3", "2026-03-03T10:00:00Z", img3, True, {"multimeter": BoundingBox(*m_box), "pcb": BoundingBox(*p_box)}))

        entities = [
            TemporalGroundTruthEntity("multimeter", [EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT], BoundingBox(*m_box), BoundingBox(*m_box)),
            TemporalGroundTruthEntity("pcb", [EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT], BoundingBox(*p_box), BoundingBox(*p_box)),
        ]
        return SyntheticTemporalSequence(
            "lighting_shifts_multi_visit",
            "Ambient illumination varies across visits without physical object movement.",
            visits,
            entities,
            seed,
        )

    def _gen_ambiguous_objects_multi_visit(self, seed: int) -> SyntheticTemporalSequence:
        """
        Sequence 7: Ambiguous, visually similar objects (e.g. twin resistor packs).
        """
        r1_box = (100, 120, 170, 210)
        r2_box = (190, 120, 260, 210)

        visits = []
        for v_idx, ts in enumerate(["2026-03-01T10:00:00Z", "2026-03-02T10:00:00Z", "2026-03-03T10:00:00Z"]):
            img = _draw_workbench_surface(self.width, self.height, seed=seed)
            _draw_resistor(img, *r1_box)
            _draw_resistor(img, *r2_box)
            visits.append(
                BenchmarkVisitFrame(
                    v_idx,
                    f"ambig_v{v_idx + 1}",
                    ts,
                    img,
                    True,
                    {"resistor_pack_1": BoundingBox(*r1_box), "resistor_pack_2": BoundingBox(*r2_box)},
                )
            )

        entities = [
            TemporalGroundTruthEntity("resistor_pack_1", [EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT], BoundingBox(*r1_box), BoundingBox(*r1_box)),
            TemporalGroundTruthEntity("resistor_pack_2", [EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT], BoundingBox(*r2_box), BoundingBox(*r2_box)),
        ]
        return SyntheticTemporalSequence(
            "ambiguous_objects_multi_visit",
            "Two identical resistor packs remain collocated across 3 visits.",
            visits,
            entities,
            seed,
        )

    def _gen_registration_failure_multi_visit(self, seed: int) -> SyntheticTemporalSequence:
        """
        Sequence 8: Registration failure in visit 2.
        Safeguard: Failed registration must not produce spatially grounded movement or removal decisions!
        """
        m_box = (280, 80, 390, 260)

        visits = []
        # Visit 1: Normal
        img1 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img1, *m_box)
        visits.append(BenchmarkVisitFrame(0, "regfail_v1", "2026-03-01T10:00:00Z", img1, True, {"multimeter": BoundingBox(*m_box)}))

        # Visit 2: Featureless query surface (registration fails!)
        img2 = np.ones((self.height, self.width, 3), dtype=np.uint8) * 230
        visits.append(BenchmarkVisitFrame(1, "regfail_v2", "2026-03-02T10:00:00Z", img2, False, {"multimeter": None}))

        # Visit 3: Texture restored, normal registration
        img3 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img3, *m_box)
        visits.append(BenchmarkVisitFrame(2, "regfail_v3", "2026-03-03T10:00:00Z", img3, True, {"multimeter": BoundingBox(*m_box)}))

        entities = [
            TemporalGroundTruthEntity(
                "multimeter",
                [EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT],
                BoundingBox(*m_box),
                BoundingBox(*m_box),
                notes="Registration fails in visit 2; engine must abstain from declaring removal; confirmed present in visit 3.",
            ),
        ]
        return SyntheticTemporalSequence(
            "registration_failure_multi_visit",
            "Registration fails in visit 2, preventing unverified removal; normal alignment restored in visit 3.",
            visits,
            entities,
            seed,
        )

    def _gen_out_of_frame_reentry(self, seed: int) -> SyntheticTemporalSequence:
        """
        Sequence 9: Object becoming visible after being outside the camera frame.
        """
        m_box = (280, 80, 390, 260)
        # Wire strippers at bottom right boundary in visit 1
        ws_box_boundary = (560, 410, 635, 475)
        # Wire strippers fully inside workbench in visits 2 and 3
        ws_box_in = (430, 280, 505, 390)

        visits = []
        # Visit 1: Multimeter inside, strippers on boundary
        img1 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img1, *m_box)
        _draw_wire_strippers(img1, *ws_box_boundary)
        visits.append(BenchmarkVisitFrame(0, "reentry_v1", "2026-03-01T10:00:00Z", img1, True, {"multimeter": BoundingBox(*m_box), "wire_stripper": BoundingBox(*ws_box_boundary)}))

        # Visit 2: Wire strippers moved fully into view
        img2 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img2, *m_box)
        _draw_wire_strippers(img2, *ws_box_in)
        visits.append(BenchmarkVisitFrame(1, "reentry_v2", "2026-03-02T10:00:00Z", img2, True, {"multimeter": BoundingBox(*m_box), "wire_stripper": BoundingBox(*ws_box_in)}))

        # Visit 3: Remains in view
        img3 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img3, *m_box)
        _draw_wire_strippers(img3, *ws_box_in)
        visits.append(BenchmarkVisitFrame(2, "reentry_v3", "2026-03-03T10:00:00Z", img3, True, {"multimeter": BoundingBox(*m_box), "wire_stripper": BoundingBox(*ws_box_in)}))

        entities = [
            TemporalGroundTruthEntity("multimeter", [EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT], BoundingBox(*m_box), BoundingBox(*m_box)),
            TemporalGroundTruthEntity("wire_stripper", [EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_MOVED, EpistemicState.CONFIRMED_MOVED], BoundingBox(*ws_box_boundary), BoundingBox(*ws_box_in)),
        ]
        return SyntheticTemporalSequence(
            "out_of_frame_reentry",
            "Tool near edge moves fully into view and maintains persistent identity.",
            visits,
            entities,
            seed,
        )

    def _gen_removed_object_returning(self, seed: int) -> SyntheticTemporalSequence:
        """
        Sequence 10: A removed object later returning.
        Progression:
        Visit 1: CONFIRMED_PRESENT
        Visit 2: PENDING_REMOVAL (vacant streak 1)
        Visit 3: CONFIRMED_REMOVED (vacant streak 2)
        Visit 4: CONFIRMED_PRESENT (re-detected, permitted by policy: CONFIRMED_REMOVED -> CONFIRMED_PRESENT)
        """
        m_box = (280, 80, 390, 260)
        p_box = (60, 60, 220, 190)

        visits = []
        # Visit 1: Present
        img1 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img1, *m_box)
        _draw_pcb(img1, *p_box)
        visits.append(BenchmarkVisitFrame(0, "return_v1", "2026-03-01T10:00:00Z", img1, True, {"multimeter": BoundingBox(*m_box), "pcb": BoundingBox(*p_box)}))

        # Visit 2: Removed (PENDING_REMOVAL)
        img2 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_pcb(img2, *p_box)
        visits.append(BenchmarkVisitFrame(1, "return_v2", "2026-03-02T10:00:00Z", img2, True, {"multimeter": None, "pcb": BoundingBox(*p_box)}))

        # Visit 3: Still removed (CONFIRMED_REMOVED)
        img3 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_pcb(img3, *p_box)
        visits.append(BenchmarkVisitFrame(2, "return_v3", "2026-03-03T10:00:00Z", img3, True, {"multimeter": None, "pcb": BoundingBox(*p_box)}))

        # Visit 4: Object returned to workbench!
        img4 = _draw_workbench_surface(self.width, self.height, seed=seed)
        _draw_multimeter(img4, *m_box)
        _draw_pcb(img4, *p_box)
        visits.append(BenchmarkVisitFrame(3, "return_v4", "2026-03-04T10:00:00Z", img4, True, {"multimeter": BoundingBox(*m_box), "pcb": BoundingBox(*p_box)}))

        entities = [
            TemporalGroundTruthEntity(
                "multimeter",
                [
                    EpistemicState.CONFIRMED_PRESENT,
                    EpistemicState.PENDING_REMOVAL,
                    EpistemicState.CONFIRMED_REMOVED,
                    EpistemicState.CONFIRMED_PRESENT,
                ],
                BoundingBox(*m_box),
                BoundingBox(*m_box),
                notes="Removed across visits 2-3; re-introduced in visit 4; restored to CONFIRMED_PRESENT.",
            ),
        ]
        return SyntheticTemporalSequence(
            "removed_object_returning",
            "Multimeter is removed, reaches CONFIRMED_REMOVED, and is subsequently re-introduced.",
            visits,
            entities,
            seed,
        )
