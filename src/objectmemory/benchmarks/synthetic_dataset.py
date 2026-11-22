"""
Synthetic workbench dataset generator for controlled benchmarking.

Produces reproducible pairs representing 8 distinct environmental conditions:
1. No environmental change (no_change).
2. One object moved (object_moved).
3. An object added (object_added).
4. An object removed (object_removed).
5. Lighting changed without object state change (lighting_change).
6. Partial occlusion (partial_occlusion).
7. Ambiguous or repeated-looking objects (ambiguous_objects).
8. Failed or unreliable alignment (unreliable_alignment).

Safeguard:
Every scenario includes explicit ground-truth labels, scenario metadata, and
an immutable is_synthetic = True flag. These synthetic benchmarks are for
controlled comparative testing and are explicitly NOT real photographs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from objectmemory.domain.models import BoundingBox
from objectmemory.domain.states import BaselineChangeType


@dataclass
class GroundTruthObject:
    """Ground truth annotation for a single physical object in a synthetic pair."""
    object_id: str
    label: str
    ref_bbox: Optional[BoundingBox]
    query_bbox: Optional[BoundingBox]
    ground_truth_change: BaselineChangeType
    is_occluded: bool = False
    is_ambiguous: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "ref_bbox": [self.ref_bbox.x_min, self.ref_bbox.y_min, self.ref_bbox.x_max, self.ref_bbox.y_max] if self.ref_bbox else None,
            "query_bbox": [self.query_bbox.x_min, self.query_bbox.y_min, self.query_bbox.x_max, self.query_bbox.y_max] if self.query_bbox else None,
            "ground_truth_change": self.ground_truth_change.value,
            "is_occluded": self.is_occluded,
            "is_ambiguous": self.is_ambiguous,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GroundTruthObject:
        ref_b = None
        if data.get("ref_bbox"):
            ref_b = BoundingBox(*data["ref_bbox"])
        query_b = None
        if data.get("query_bbox"):
            query_b = BoundingBox(*data["query_bbox"])
        return cls(
            object_id=data["object_id"],
            label=data["label"],
            ref_bbox=ref_b,
            query_bbox=query_b,
            ground_truth_change=BaselineChangeType(data["ground_truth_change"]),
            is_occluded=data.get("is_occluded", False),
            is_ambiguous=data.get("is_ambiguous", False),
            notes=data.get("notes", ""),
        )


@dataclass
class SyntheticScenarioPair:
    """A controlled reference-query synthetic pair with metadata and ground-truth objects."""
    scenario_name: str
    description: str
    ref_image: np.ndarray
    query_image: np.ndarray
    objects: List[GroundTruthObject]
    random_seed: int
    is_synthetic: bool = True
    expected_alignment_success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "description": self.description,
            "is_synthetic": True,
            "random_seed": self.random_seed,
            "expected_alignment_success": self.expected_alignment_success,
            "objects": [obj.to_dict() for obj in self.objects],
            "metadata": self.metadata,
        }


# =============================================================================
# Drawing Helpers for Synthetic Workbench Visuals
# =============================================================================

def _draw_workbench_surface(width: int = 640, height: int = 480, seed: int = 42) -> np.ndarray:
    """Draws a dark-gray anti-static workbench mat with grid lines and fiducials."""
    rng = np.random.RandomState(seed)
    img = np.ones((height, width, 3), dtype=np.uint8) * 45  # Dark mat

    # Grid lines (25px spacing)
    for x in range(0, width, 25):
        cv2.line(img, (x, 0), (x, height), (75, 75, 75), 1)
    for y in range(0, height, 25):
        cv2.line(img, (0, y), (width, y), (75, 75, 75), 1)

    # Scattered solder specks / fiducials for high-contrast SIFT keypoints
    for _ in range(40):
        cx = rng.randint(15, width - 15)
        cy = rng.randint(15, height - 15)
        radius = rng.randint(2, 5)
        color = (int(rng.randint(160, 240)), int(rng.randint(160, 240)), int(rng.randint(160, 240)))
        cv2.circle(img, (cx, cy), radius, color, -1)
        cv2.circle(img, (cx, cy), radius + 1, (10, 10, 10), 1)

    return img


def _draw_multimeter(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    """Draws a bright yellow handheld multimeter with a LCD display and rotary dial."""
    cv2.rectangle(img, (x0, y0), (x1, y1), (25, 200, 245), -1)      # Yellow rubber holster
    cv2.rectangle(img, (x0, y0), (x1, y1), (10, 10, 10), 2)
    # LCD Display
    dw = int((x1 - x0) * 0.7)
    dh = int((y1 - y0) * 0.25)
    dx0 = x0 + int((x1 - x0) * 0.15)
    dy0 = y0 + int((y1 - y0) * 0.1)
    cv2.rectangle(img, (dx0, dy0), (dx0 + dw, dy0 + dh), (60, 80, 60), -1)
    cv2.rectangle(img, (dx0, dy0), (dx0 + dw, dy0 + dh), (180, 220, 180), 1)
    # Rotary Dial
    dial_cx = (x0 + x1) // 2
    dial_cy = y0 + int((y1 - y0) * 0.65)
    dial_r = max(5, int((x1 - x0) * 0.22))
    cv2.circle(img, (dial_cx, dial_cy), dial_r, (30, 30, 30), -1)
    cv2.circle(img, (dial_cx, dial_cy), dial_r, (200, 200, 200), 1)


def _draw_pcb(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    """Draws a green printed circuit board with solder pads and chip package."""
    cv2.rectangle(img, (x0, y0), (x1, y1), (35, 130, 45), -1)       # Green solder mask
    cv2.rectangle(img, (x0, y0), (x1, y1), (210, 210, 210), 2)
    # Centered IC chip
    cx0 = x0 + int((x1 - x0) * 0.25)
    cy0 = y0 + int((y1 - y0) * 0.25)
    cx1 = x1 - int((x1 - x0) * 0.25)
    cy1 = y1 - int((y1 - y0) * 0.25)
    cv2.rectangle(img, (cx0, cy0), (cx1, cy1), (20, 20, 20), -1)
    # Silver pin tracks
    for px in range(cx0, cx1, max(4, (cx1 - cx0) // 5)):
        cv2.line(img, (px, cy0 - 4), (px, cy0), (220, 200, 70), 1)
        cv2.line(img, (px, cy1), (px, cy1 + 4), (220, 200, 70), 1)


def _draw_component_box(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    """Draws a translucent blue plastic component organizer box."""
    cv2.rectangle(img, (x0, y0), (x1, y1), (180, 90, 30), -1)       # Blue body
    cv2.rectangle(img, (x0, y0), (x1, y1), (240, 240, 240), 2)
    mid_x = (x0 + x1) // 2
    mid_y = (y0 + y1) // 2
    cv2.line(img, (mid_x, y0), (mid_x, y1), (200, 120, 50), 1)
    cv2.line(img, (x0, mid_y), (x1, mid_y), (200, 120, 50), 1)


def _draw_wire_strippers(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    """Draws red-handled wire strippers with metallic jaws."""
    # Handle
    cv2.rectangle(img, (x0, y0 + (y1 - y0) // 3), (x1, y1), (30, 30, 210), -1)
    # Jaws
    cv2.rectangle(img, (x0 + (x1 - x0) // 4, y0), (x1 - (x1 - x0) // 4, y0 + (y1 - y0) // 3), (180, 180, 180), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (10, 10, 10), 1)


def _draw_resistor(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    """Draws a small beige axial resistor with colored bands."""
    cv2.rectangle(img, (x0, y0), (x1, y1), (140, 180, 210), -1)     # Beige body
    cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 20), 1)
    bw = max(2, (x1 - x0) // 6)
    # Brown, Black, Red bands
    cv2.rectangle(img, (x0 + bw, y0), (x0 + 2 * bw, y1), (30, 50, 100), -1)
    cv2.rectangle(img, (x0 + 2 * bw + 2, y0), (x0 + 3 * bw + 2, y1), (10, 10, 10), -1)
    cv2.rectangle(img, (x0 + 3 * bw + 4, y0), (x0 + 4 * bw + 4, y1), (20, 20, 190), -1)


# =============================================================================
# Generator Class
# =============================================================================

class SyntheticWorkbenchDatasetGenerator:
    """
    Generates reproducible synthetic workbench pairs for the 8 controlled scenarios.
    """

    SUPPORTED_SCENARIOS = [
        "no_change",
        "object_moved",
        "object_added",
        "object_removed",
        "lighting_change",
        "partial_occlusion",
        "ambiguous_objects",
        "unreliable_alignment",
    ]

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height

    def generate_all_scenarios(self, base_seed: int = 42) -> List[SyntheticScenarioPair]:
        """Generates all 8 benchmark scenarios deterministically."""
        return [
            self.generate_scenario(scenario_name, seed=base_seed + idx * 17)
            for idx, scenario_name in enumerate(self.SUPPORTED_SCENARIOS)
        ]

    def generate_scenario(self, scenario_name: str, seed: int = 42) -> SyntheticScenarioPair:
        """Generates a single controlled scenario by name."""
        if scenario_name == "no_change":
            return self._gen_no_change(seed)
        elif scenario_name == "object_moved":
            return self._gen_object_moved(seed)
        elif scenario_name == "object_added":
            return self._gen_object_added(seed)
        elif scenario_name == "object_removed":
            return self._gen_object_removed(seed)
        elif scenario_name == "lighting_change":
            return self._gen_lighting_change(seed)
        elif scenario_name == "partial_occlusion":
            return self._gen_partial_occlusion(seed)
        elif scenario_name == "ambiguous_objects":
            return self._gen_ambiguous_objects(seed)
        elif scenario_name == "unreliable_alignment":
            return self._gen_unreliable_alignment(seed)
        else:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. Supported: {self.SUPPORTED_SCENARIOS}"
            )

    # -------------------------------------------------------------------------
    # Scenario Implementations
    # -------------------------------------------------------------------------

    def _gen_no_change(self, seed: int) -> SyntheticScenarioPair:
        """Scenario 1: No environmental change."""
        ref = _draw_workbench_surface(self.width, self.height, seed=seed)
        query = _draw_workbench_surface(self.width, self.height, seed=seed)

        # Object 1: Multimeter
        m_box = (280, 80, 390, 260)
        _draw_multimeter(ref, *m_box)
        _draw_multimeter(query, *m_box)

        # Object 2: PCB
        p_box = (60, 60, 220, 190)
        _draw_pcb(ref, *p_box)
        _draw_pcb(query, *p_box)

        # Object 3: Component Box
        c_box = (140, 280, 270, 390)
        _draw_component_box(ref, *c_box)
        _draw_component_box(query, *c_box)

        objects = [
            GroundTruthObject("obj_multimeter", "multimeter", BoundingBox(*m_box), BoundingBox(*m_box), BaselineChangeType.UNCHANGED),
            GroundTruthObject("obj_pcb", "pcb", BoundingBox(*p_box), BoundingBox(*p_box), BaselineChangeType.UNCHANGED),
            GroundTruthObject("obj_compbox", "component_box", BoundingBox(*c_box), BoundingBox(*c_box), BaselineChangeType.UNCHANGED),
        ]
        return SyntheticScenarioPair(
            scenario_name="no_change",
            description="Workbench observed twice with identical objects and positions.",
            ref_image=ref,
            query_image=query,
            objects=objects,
            random_seed=seed,
        )

    def _gen_object_moved(self, seed: int) -> SyntheticScenarioPair:
        """Scenario 2: One object moved."""
        ref = _draw_workbench_surface(self.width, self.height, seed=seed)
        query = _draw_workbench_surface(self.width, self.height, seed=seed)

        # Object 1: Multimeter (MOVED from (280, 80) to (440, 220))
        m_ref_box = (280, 80, 390, 260)
        m_qry_box = (440, 220, 550, 400)
        _draw_multimeter(ref, *m_ref_box)
        _draw_multimeter(query, *m_qry_box)

        # Object 2: PCB (UNCHANGED)
        p_box = (60, 60, 220, 190)
        _draw_pcb(ref, *p_box)
        _draw_pcb(query, *p_box)

        objects = [
            GroundTruthObject("obj_multimeter", "multimeter", BoundingBox(*m_ref_box), BoundingBox(*m_qry_box), BaselineChangeType.MOVED, notes="Relocated by ~210px"),
            GroundTruthObject("obj_pcb", "pcb", BoundingBox(*p_box), BoundingBox(*p_box), BaselineChangeType.UNCHANGED),
        ]
        return SyntheticScenarioPair(
            scenario_name="object_moved",
            description="Workbench where one multimeter has been moved to a new position.",
            ref_image=ref,
            query_image=query,
            objects=objects,
            random_seed=seed,
        )

    def _gen_object_added(self, seed: int) -> SyntheticScenarioPair:
        """Scenario 3: An object added."""
        ref = _draw_workbench_surface(self.width, self.height, seed=seed)
        query = _draw_workbench_surface(self.width, self.height, seed=seed)

        # Object 1: Multimeter (UNCHANGED)
        m_box = (280, 80, 390, 260)
        _draw_multimeter(ref, *m_box)
        _draw_multimeter(query, *m_box)

        # Object 2: PCB (UNCHANGED)
        p_box = (60, 60, 220, 190)
        _draw_pcb(ref, *p_box)
        _draw_pcb(query, *p_box)

        # Object 3: Wire strippers (ADDED in query only)
        w_box = (420, 260, 540, 370)
        _draw_wire_strippers(query, *w_box)

        objects = [
            GroundTruthObject("obj_multimeter", "multimeter", BoundingBox(*m_box), BoundingBox(*m_box), BaselineChangeType.UNCHANGED),
            GroundTruthObject("obj_pcb", "pcb", BoundingBox(*p_box), BoundingBox(*p_box), BaselineChangeType.UNCHANGED),
            GroundTruthObject("obj_strippers", "wire_strippers", None, BoundingBox(*w_box), BaselineChangeType.ADDED, notes="Newly placed tool"),
        ]
        return SyntheticScenarioPair(
            scenario_name="object_added",
            description="Workbench where a new tool (wire strippers) has been placed.",
            ref_image=ref,
            query_image=query,
            objects=objects,
            random_seed=seed,
        )

    def _gen_object_removed(self, seed: int) -> SyntheticScenarioPair:
        """Scenario 4: An object removed."""
        ref = _draw_workbench_surface(self.width, self.height, seed=seed)
        query = _draw_workbench_surface(self.width, self.height, seed=seed)

        # Object 1: Multimeter (UNCHANGED)
        m_box = (280, 80, 390, 260)
        _draw_multimeter(ref, *m_box)
        _draw_multimeter(query, *m_box)

        # Object 2: Component Box (REMOVED: drawn in ref, vacant in query)
        c_box = (140, 280, 270, 390)
        _draw_component_box(ref, *c_box)

        objects = [
            GroundTruthObject("obj_multimeter", "multimeter", BoundingBox(*m_box), BoundingBox(*m_box), BaselineChangeType.UNCHANGED),
            GroundTruthObject("obj_compbox", "component_box", BoundingBox(*c_box), None, BaselineChangeType.REMOVED, notes="Object removed, vacant mat remaining"),
        ]
        return SyntheticScenarioPair(
            scenario_name="object_removed",
            description="Workbench where one component box has been completely removed.",
            ref_image=ref,
            query_image=query,
            objects=objects,
            random_seed=seed,
        )

    def _gen_lighting_change(self, seed: int) -> SyntheticScenarioPair:
        """Scenario 5: Lighting changed without an object-state change."""
        ref = _draw_workbench_surface(self.width, self.height, seed=seed)
        query = _draw_workbench_surface(self.width, self.height, seed=seed)

        m_box = (280, 80, 390, 260)
        p_box = (60, 60, 220, 190)
        _draw_multimeter(ref, *m_box)
        _draw_multimeter(query, *m_box)
        _draw_pcb(ref, *p_box)
        _draw_pcb(query, *p_box)

        # Simulate warm/dimmed room lighting on query visit:
        # Scale brightness down to 55% and add an amber color shift
        query = (query.astype(np.float32) * np.array([0.55, 0.65, 0.75])).clip(0, 255).astype(np.uint8)

        objects = [
            GroundTruthObject("obj_multimeter", "multimeter", BoundingBox(*m_box), BoundingBox(*m_box), BaselineChangeType.UNCHANGED, notes="Lighting dimmed 45%"),
            GroundTruthObject("obj_pcb", "pcb", BoundingBox(*p_box), BoundingBox(*p_box), BaselineChangeType.UNCHANGED, notes="Lighting dimmed 45%"),
        ]
        return SyntheticScenarioPair(
            scenario_name="lighting_change",
            description="Workbench with identical objects under significantly altered illumination.",
            ref_image=ref,
            query_image=query,
            objects=objects,
            random_seed=seed,
            metadata={"lighting_scaled": True},
        )

    def _gen_partial_occlusion(self, seed: int) -> SyntheticScenarioPair:
        """Scenario 6: Partial occlusion."""
        ref = _draw_workbench_surface(self.width, self.height, seed=seed)
        query = _draw_workbench_surface(self.width, self.height, seed=seed)

        m_box = (280, 80, 390, 260)
        p_box = (60, 60, 220, 190)
        _draw_multimeter(ref, *m_box)
        _draw_multimeter(query, *m_box)
        _draw_pcb(ref, *p_box)
        _draw_pcb(query, *p_box)

        # Occlude 55% of the multimeter in query image with a simulated manual/clipboard
        # Multimeter is (280, 80, 390, 260). Occlude y from 160 to 260.
        cv2.rectangle(query, (260, 160), (410, 280), (220, 210, 190), -1)  # Beige clipboard
        cv2.rectangle(query, (260, 160), (410, 280), (10, 10, 10), 2)

        objects = [
            GroundTruthObject("obj_multimeter", "multimeter", BoundingBox(*m_box), BoundingBox(*m_box), BaselineChangeType.UNCHANGED, is_occluded=True, notes="55% occluded by clipboard"),
            GroundTruthObject("obj_pcb", "pcb", BoundingBox(*p_box), BoundingBox(*p_box), BaselineChangeType.UNCHANGED),
        ]
        return SyntheticScenarioPair(
            scenario_name="partial_occlusion",
            description="Workbench where one object is partially occluded by another surface.",
            ref_image=ref,
            query_image=query,
            objects=objects,
            random_seed=seed,
        )

    def _gen_ambiguous_objects(self, seed: int) -> SyntheticScenarioPair:
        """Scenario 7: Ambiguous or repeated-looking objects."""
        ref = _draw_workbench_surface(self.width, self.height, seed=seed)
        query = _draw_workbench_surface(self.width, self.height, seed=seed)

        # Two identical resistors placed next to each other
        r1_box = (120, 120, 190, 160)
        r2_box = (230, 120, 300, 160)
        _draw_resistor(ref, *r1_box)
        _draw_resistor(ref, *r2_box)
        _draw_resistor(query, *r1_box)
        _draw_resistor(query, *r2_box)

        # Unique multimeter
        m_box = (380, 80, 490, 260)
        _draw_multimeter(ref, *m_box)
        _draw_multimeter(query, *m_box)

        objects = [
            GroundTruthObject("obj_resistor_1", "resistor", BoundingBox(*r1_box), BoundingBox(*r1_box), BaselineChangeType.UNCHANGED, is_ambiguous=True, notes="Identical twin resistor"),
            GroundTruthObject("obj_resistor_2", "resistor", BoundingBox(*r2_box), BoundingBox(*r2_box), BaselineChangeType.UNCHANGED, is_ambiguous=True, notes="Identical twin resistor"),
            GroundTruthObject("obj_multimeter", "multimeter", BoundingBox(*m_box), BoundingBox(*m_box), BaselineChangeType.UNCHANGED),
        ]
        return SyntheticScenarioPair(
            scenario_name="ambiguous_objects",
            description="Workbench containing identical, visually ambiguous components.",
            ref_image=ref,
            query_image=query,
            objects=objects,
            random_seed=seed,
        )

    def _gen_unreliable_alignment(self, seed: int) -> SyntheticScenarioPair:
        """Scenario 8: Failed or unreliable alignment."""
        ref = _draw_workbench_surface(self.width, self.height, seed=seed)

        # Query is a completely featureless / blank white blur with near-zero keypoints
        query = np.ones((self.height, self.width, 3), dtype=np.uint8) * 230

        m_box = (280, 80, 390, 260)
        _draw_multimeter(ref, *m_box)

        objects = [
            GroundTruthObject("obj_multimeter", "multimeter", BoundingBox(*m_box), None, BaselineChangeType.UNKNOWN, notes="Alignment fails; ground truth status in query is unverifiable"),
        ]
        return SyntheticScenarioPair(
            scenario_name="unreliable_alignment",
            description="Registration failure test case: query surface lacks trackable features.",
            ref_image=ref,
            query_image=query,
            objects=objects,
            random_seed=seed,
            expected_alignment_success=False,
        )

    # -------------------------------------------------------------------------
    # Disk Persistence
    # -------------------------------------------------------------------------

    def save_dataset(self, output_dir: Path, pairs: Optional[List[SyntheticScenarioPair]] = None) -> None:
        """Saves generated scenario pairs and annotations to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if pairs is None:
            pairs = self.generate_all_scenarios()

        manifest = []
        for pair in pairs:
            scenario_dir = output_dir / pair.scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)

            ref_path = scenario_dir / "ref.png"
            query_path = scenario_dir / "query.png"
            meta_path = scenario_dir / "ground_truth.json"

            cv2.imwrite(str(ref_path), pair.ref_image)
            cv2.imwrite(str(query_path), pair.query_image)

            meta = pair.to_metadata_dict()
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            manifest.append({
                "scenario_name": pair.scenario_name,
                "description": pair.description,
                "objects_count": len(pair.objects),
                "path": str(scenario_dir.relative_to(output_dir)),
            })

        with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump({"scenarios": manifest, "is_synthetic": True}, f, indent=2)

    def load_dataset(self, input_dir: Path) -> List[SyntheticScenarioPair]:
        """Loads scenario pairs and annotations from disk."""
        input_dir = Path(input_dir)
        manifest_file = input_dir / "manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found in {input_dir}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        pairs: List[SyntheticScenarioPair] = []
        for item in manifest.get("scenarios", []):
            scenario_name = item["scenario_name"]
            scenario_dir = input_dir / scenario_name

            ref_img = cv2.imread(str(scenario_dir / "ref.png"))
            query_img = cv2.imread(str(scenario_dir / "query.png"))
            if ref_img is None or query_img is None:
                raise IOError(f"Could not read images for scenario {scenario_name}")

            with open(scenario_dir / "ground_truth.json", "r", encoding="utf-8") as f:
                meta = json.load(f)

            objects = [GroundTruthObject.from_dict(obj) for obj in meta["objects"]]
            pairs.append(
                SyntheticScenarioPair(
                    scenario_name=scenario_name,
                    description=meta["description"],
                    ref_image=ref_img,
                    query_image=query_img,
                    objects=objects,
                    random_seed=meta["random_seed"],
                    is_synthetic=True,
                    expected_alignment_success=meta.get("expected_alignment_success", True),
                    metadata=meta.get("metadata", {}),
                )
            )

        return pairs
