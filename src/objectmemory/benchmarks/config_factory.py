"""
Benchmark Configuration Factory for ObjectMemory.

Provides immutable, auditable, and deterministic configuration objects for
B0–B3 baselines and A1–A6 ablations. Enforces strict scientific integrity:
unsupported configurations are explicitly classified as BLOCKED and rejected
rather than being faked through misleading configuration labels.

Guiding Principles:
1. Production Isolation: Benchmark configurations never mutate global or production defaults.
2. Independent Instantiation: Every configuration call returns fresh, decoupled instances.
3. Complete Parameter Transparency: Records all effective parameters, not just overrides.
4. Deterministic Serialization: Provides stable JSON serialization and cryptographic hashes.
5. Strict Feasibility Gating: Configurations that cannot be genuinely executed by the current
   engine without modifying production code raise UnsupportedConfigurationError.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from objectmemory.temporal.models import TemporalEngineConfig


class ConfigurationSupportStatus(str, Enum):
    """Operational feasibility of a benchmark configuration in the current engine."""
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    BLOCKED = "BLOCKED"


class UnsupportedConfigurationError(ValueError):
    """Raised when an unsupported or blocked baseline/ablation configuration is requested."""
    pass


class InvalidConfigurationCombinationError(ValueError):
    """Raised when contradictory or invalid configuration parameters are combined."""
    pass


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Immutable representation of an experimental baseline or ablation configuration.
    """
    config_id: str
    name: str
    description: str
    support_status: ConfigurationSupportStatus
    pipeline_type: str  # "FULL_TEMPORAL_MEMORY" or "MEMORYLESS_DETECTION"
    temporal_config: Optional[TemporalEngineConfig]
    enabled_safeguards: Tuple[str, ...]
    disabled_safeguards: Tuple[str, ...]
    notes: str
    effective_parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Returns deterministic dictionary representation."""
        return {
            "config_id": self.config_id,
            "name": self.name,
            "description": self.description,
            "support_status": self.support_status.value,
            "pipeline_type": self.pipeline_type,
            "enabled_safeguards": list(self.enabled_safeguards),
            "disabled_safeguards": list(self.disabled_safeguards),
            "notes": self.notes,
            "effective_parameters": self.effective_parameters,
        }

    def to_json(self, indent: int = 2) -> str:
        """Returns deterministic, sorted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @property
    def deterministic_hash(self) -> str:
        """SHA-256 fingerprint computed from the deterministic JSON serialization."""
        serialized = self.to_json(indent=0)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get_isolated_temporal_config(self) -> Optional[TemporalEngineConfig]:
        """Returns a safe, decoupled copy of the underlying TemporalEngineConfig."""
        if self.temporal_config is None:
            return None
        return copy.deepcopy(self.temporal_config)


class BenchmarkConfigFactory:
    """
    Factory constructing validated, isolated configurations for ObjectMemory benchmarks.
    """

    KNOWN_CONFIG_IDS: Set[str] = {
        "B0", "B1", "B2", "B3",
        "A1", "A2", "A3", "A4", "A5", "A6",
    }

    @classmethod
    def list_known_configs(cls) -> List[str]:
        """Returns sorted list of all recognized benchmark configuration IDs."""
        return sorted(cls.KNOWN_CONFIG_IDS)

    @classmethod
    def get_config_status(cls, config_id: str) -> ConfigurationSupportStatus:
        """Returns the support status for a configuration without instantiating it."""
        cid = config_id.strip().upper()
        if cid not in cls.KNOWN_CONFIG_IDS:
            raise ValueError(f"Unknown benchmark configuration ID: '{config_id}'. Known: {cls.list_known_configs()}")

        if cid in {"B3", "B0", "A5"}:
            return ConfigurationSupportStatus.SUPPORTED
        if cid in {"A2", "A3"}:
            return ConfigurationSupportStatus.PARTIALLY_SUPPORTED
        return ConfigurationSupportStatus.BLOCKED

    @classmethod
    def create_config(
        cls,
        config_id: str,
        allow_blocked: bool = False,
        allow_partial: bool = False,
    ) -> BenchmarkConfig:
        """
        Instantiates an isolated, immutable BenchmarkConfig.

        Args:
            config_id: One of B0..B3 or A1..A6.
            allow_blocked: If True, permits returning BLOCKED configurations for audit/documentation.
                           Defaults to False (raises UnsupportedConfigurationError).
            allow_partial: If True, permits returning PARTIALLY_SUPPORTED configurations.
                           Defaults to False (raises UnsupportedConfigurationError).

        Raises:
            ValueError: If config_id is unrecognized.
            UnsupportedConfigurationError: If the requested configuration is BLOCKED or PARTIALLY_SUPPORTED
                                           and the corresponding opt-in flag is False.
        """
        cid = config_id.strip().upper()
        if cid not in cls.KNOWN_CONFIG_IDS:
            raise ValueError(
                f"Unknown benchmark configuration ID '{config_id}'. "
                f"Supported identifiers: {cls.list_known_configs()}"
            )

        status = cls.get_config_status(cid)
        if status == ConfigurationSupportStatus.BLOCKED and not allow_blocked:
            reason = cls._get_block_reason(cid)
            raise UnsupportedConfigurationError(
                f"Configuration '{cid}' is BLOCKED in the current engine: {reason} "
                f"Pass allow_blocked=True only if inspecting theoretical specifications for auditing."
            )

        if status == ConfigurationSupportStatus.PARTIALLY_SUPPORTED and not allow_partial:
            reason = cls._get_partial_reason(cid)
            raise UnsupportedConfigurationError(
                f"Configuration '{cid}' is only PARTIALLY_SUPPORTED in the current engine: {reason} "
                f"Pass allow_partial=True to instantiate with documented caveats."
            )

        return cls._build_config(cid, status)

    @classmethod
    def get_supported_configs(cls) -> Dict[str, BenchmarkConfig]:
        """Returns dictionary of all natively supported configurations (B0, B3, A5)."""
        supported_ids = ["B0", "B3", "A5"]
        return {cid: cls.create_config(cid) for cid in supported_ids}

    # -------------------------------------------------------------------------
    # Internal Builders
    # -------------------------------------------------------------------------

    @classmethod
    def _build_config(cls, cid: str, status: ConfigurationSupportStatus) -> BenchmarkConfig:
        if cid == "B0":
            return cls._build_b0(status)
        elif cid == "B1":
            return cls._build_b1(status)
        elif cid == "B2":
            return cls._build_b2(status)
        elif cid == "B3":
            return cls._build_b3(status)
        elif cid == "A1":
            return cls._build_a1(status)
        elif cid == "A2":
            return cls._build_a2(status)
        elif cid == "A3":
            return cls._build_a3(status)
        elif cid == "A4":
            return cls._build_a4(status)
        elif cid == "A5":
            return cls._build_a5(status)
        elif cid == "A6":
            return cls._build_a6(status)
        raise ValueError(f"Unhandled builder for: {cid}")

    @classmethod
    def _build_b0(cls, status: ConfigurationSupportStatus) -> BenchmarkConfig:
        params = {
            "pipeline_type": "MEMORYLESS_DETECTION",
            "detector": "YOLOv8n-ONNX",
            "confidence_threshold": 0.35,
            "nms_threshold": 0.45,
            "temporal_memory": False,
            "cross_visit_association": False,
            "state_transitions": False,
        }
        return BenchmarkConfig(
            config_id="B0",
            name="Memoryless Detector Reference",
            description="Per-frame independent YOLOv8n detections without cross-session temporal memory.",
            support_status=status,
            pipeline_type="MEMORYLESS_DETECTION",
            temporal_config=None,
            enabled_safeguards=(),
            disabled_safeguards=(
                "registration_absence_freezing",
                "appearance_vacancy_verification",
                "visibility_fov_safeguards",
                "multivisit_removal_streak",
                "ambiguity_gating",
                "historical_identity_recovery",
            ),
            notes="Evaluated strictly per-frame against ground truth. Does not execute TemporalMemoryEngine.",
            effective_parameters=params,
        )

    @classmethod
    def _build_b1(cls, status: ConfigurationSupportStatus) -> BenchmarkConfig:
        params = {
            "pipeline_type": "CONVENTIONAL_HUNGARIAN_TRACKING",
            "detector": "YOLOv8n-ONNX",
            "matching_cost": "1.0 - (0.50 * IoU + 0.50 * appearance_sim)",
            "ambiguity_margin": 0.0,
            "disappearance_policy": "IMMEDIATE_REMOVAL",
            "temporal_safeguards": False,
        }
        return BenchmarkConfig(
            config_id="B1",
            name="Conventional Hungarian Tracking",
            description="Standard detector-based Hungarian association with immediate disappearance removal.",
            support_status=status,
            pipeline_type="CONVENTIONAL_HUNGARIAN_TRACKING",
            temporal_config=None,
            enabled_safeguards=(),
            disabled_safeguards=(
                "registration_absence_freezing",
                "appearance_vacancy_verification",
                "visibility_fov_safeguards",
                "multivisit_removal_streak",
                "ambiguity_gating",
                "historical_identity_recovery",
            ),
            notes=cls._get_block_reason("B1"),
            effective_parameters=params,
        )

    @classmethod
    def _build_b2(cls, status: ConfigurationSupportStatus) -> BenchmarkConfig:
        params = {
            "pipeline_type": "UNGATED_TEMPORAL_MEMORY",
            "registration_absence_freezing": False,
            "appearance_vacancy_verification": False,
            "visibility_fov_safeguards": False,
            "occlusion_safeguards": False,
        }
        return BenchmarkConfig(
            config_id="B2",
            name="Ungated Temporal Memory",
            description="Multi-visit temporal memory without registration-aware, visibility, or vacancy safeguards.",
            support_status=status,
            pipeline_type="UNGATED_TEMPORAL_MEMORY",
            temporal_config=None,
            enabled_safeguards=("multivisit_state_transitions",),
            disabled_safeguards=(
                "registration_absence_freezing",
                "appearance_vacancy_verification",
                "visibility_fov_safeguards",
                "occlusion_gating",
            ),
            notes=cls._get_block_reason("B2"),
            effective_parameters=params,
        )

    @classmethod
    def _build_b3(cls, status: ConfigurationSupportStatus) -> BenchmarkConfig:
        cfg = TemporalEngineConfig()  # Production defaults
        params = cls._extract_temporal_params(cfg)
        return BenchmarkConfig(
            config_id="B3",
            name="Complete ObjectMemory",
            description="Proposed system with uncertainty-aware, registration-gated, vacancy-verified memory.",
            support_status=status,
            pipeline_type="FULL_TEMPORAL_MEMORY",
            temporal_config=cfg,
            enabled_safeguards=(
                "registration_absence_freezing",
                "appearance_vacancy_verification",
                "visibility_fov_safeguards",
                "occlusion_gating",
                "multivisit_removal_streak",
                "ambiguity_gating",
                "historical_identity_recovery",
                "dynamic_uncertainty_budget",
                "drift_mitigation",
                "boundary_truncation_handling",
            ),
            disabled_safeguards=(),
            notes="Matches production configuration defaults exactly.",
            effective_parameters=params,
        )

    @classmethod
    def _build_a1(cls, status: ConfigurationSupportStatus) -> BenchmarkConfig:
        cfg = TemporalEngineConfig(min_registration_inlier_ratio=0.0)
        params = cls._extract_temporal_params(cfg)
        params["override_target"] = "disable_registration_absence_freezing"
        return BenchmarkConfig(
            config_id="A1",
            name="Ablation A1: No Registration Absence Freezing",
            description="Allows absence reasoning to proceed on unaligned or poorly aligned frames.",
            support_status=status,
            pipeline_type="FULL_TEMPORAL_MEMORY",
            temporal_config=cfg,
            enabled_safeguards=(
                "appearance_vacancy_verification",
                "visibility_fov_safeguards",
                "multivisit_removal_streak",
                "ambiguity_gating",
                "historical_identity_recovery",
            ),
            disabled_safeguards=("registration_absence_freezing",),
            notes=cls._get_block_reason("A1"),
            effective_parameters=params,
        )

    @classmethod
    def _build_a2(cls, status: ConfigurationSupportStatus) -> BenchmarkConfig:
        # vacancy_similarity_max = 1.0 forces any patch cosine similarity < 1.0 to evaluate as vacant
        cfg = TemporalEngineConfig(vacancy_similarity_max=1.0)
        params = cls._extract_temporal_params(cfg)
        params["override_target"] = "disable_appearance_vacancy_verification"
        return BenchmarkConfig(
            config_id="A2",
            name="Ablation A2: No Appearance Vacancy Verification",
            description="Bypasses visual patch vacancy checks; treats unobserved detections as vacant.",
            support_status=status,
            pipeline_type="FULL_TEMPORAL_MEMORY",
            temporal_config=cfg,
            enabled_safeguards=(
                "registration_absence_freezing",
                "visibility_fov_safeguards",
                "multivisit_removal_streak",
                "ambiguity_gating",
                "historical_identity_recovery",
            ),
            disabled_safeguards=("appearance_vacancy_verification",),
            notes=cls._get_partial_reason("A2"),
            effective_parameters=params,
        )

    @classmethod
    def _build_a3(cls, status: ConfigurationSupportStatus) -> BenchmarkConfig:
        # min_overlap_fraction = 0.0 disables bounding-box FOV clipping
        cfg = TemporalEngineConfig(min_overlap_fraction=0.0)
        params = cls._extract_temporal_params(cfg)
        params["override_target"] = "disable_fov_safeguard"
        return BenchmarkConfig(
            config_id="A3",
            name="Ablation A3: No Visibility & FOV Safeguards",
            description="Bypasses camera field-of-view overlap gating on canonical bounding boxes.",
            support_status=status,
            pipeline_type="FULL_TEMPORAL_MEMORY",
            temporal_config=cfg,
            enabled_safeguards=(
                "registration_absence_freezing",
                "appearance_vacancy_verification",
                "multivisit_removal_streak",
                "ambiguity_gating",
                "historical_identity_recovery",
            ),
            disabled_safeguards=("visibility_fov_safeguards",),
            notes=cls._get_partial_reason("A3"),
            effective_parameters=params,
        )

    @classmethod
    def _build_a4(cls, status: ConfigurationSupportStatus) -> BenchmarkConfig:
        cfg = TemporalEngineConfig(vacancy_visits_for_removal=1)
        params = cls._extract_temporal_params(cfg)
        params["override_target"] = "single_visit_removal_confirmation"
        return BenchmarkConfig(
            config_id="A4",
            name="Ablation A4: Single-Visit Removal Confirmation",
            description="Attempts removal confirmation after a single unobserved visit.",
            support_status=status,
            pipeline_type="FULL_TEMPORAL_MEMORY",
            temporal_config=cfg,
            enabled_safeguards=(
                "registration_absence_freezing",
                "appearance_vacancy_verification",
                "visibility_fov_safeguards",
                "ambiguity_gating",
                "historical_identity_recovery",
            ),
            disabled_safeguards=("multivisit_removal_streak",),
            notes=cls._get_block_reason("A4"),
            effective_parameters=params,
        )

    @classmethod
    def _build_a5(cls, status: ConfigurationSupportStatus) -> BenchmarkConfig:
        # Setting ambiguity_margin = 0.0 natively disables ambiguity rejection in associator.py
        cfg = TemporalEngineConfig(ambiguity_margin=0.0)
        params = cls._extract_temporal_params(cfg)
        params["override_target"] = "disable_ambiguity_gating"
        return BenchmarkConfig(
            config_id="A5",
            name="Ablation A5: No Ambiguity Handling",
            description="Disables candidate ambiguity margin, forcing one-to-one assignment to top candidate.",
            support_status=status,
            pipeline_type="FULL_TEMPORAL_MEMORY",
            temporal_config=cfg,
            enabled_safeguards=(
                "registration_absence_freezing",
                "appearance_vacancy_verification",
                "visibility_fov_safeguards",
                "multivisit_removal_streak",
                "historical_identity_recovery",
            ),
            disabled_safeguards=("ambiguity_gating",),
            notes="Fully supported natively via ambiguity_margin=0.0 in TemporalEngineConfig.",
            effective_parameters=params,
        )

    @classmethod
    def _build_a6(cls, status: ConfigurationSupportStatus) -> BenchmarkConfig:
        cfg = TemporalEngineConfig()
        params = cls._extract_temporal_params(cfg)
        params["override_target"] = "disable_historical_identity_recovery"
        return BenchmarkConfig(
            config_id="A6",
            name="Ablation A6: No Historical Identity Recovery",
            description="Treats re-appearing objects as brand-new entities rather than recovering historical tracks.",
            support_status=status,
            pipeline_type="FULL_TEMPORAL_MEMORY",
            temporal_config=cfg,
            enabled_safeguards=(
                "registration_absence_freezing",
                "appearance_vacancy_verification",
                "visibility_fov_safeguards",
                "multivisit_removal_streak",
                "ambiguity_gating",
            ),
            disabled_safeguards=("historical_identity_recovery",),
            notes=cls._get_block_reason("A6"),
            effective_parameters=params,
        )

    @staticmethod
    def _extract_temporal_params(cfg: TemporalEngineConfig) -> Dict[str, Any]:
        """Extracts complete primitive parameters from TemporalEngineConfig."""
        d = copy.deepcopy(asdict(cfg))
        return d

    @staticmethod
    def _get_block_reason(cid: str) -> str:
        reasons = {
            "B1": "VALID_STATE_TRANSITIONS in domain/states.py:77 and engine.py:1070 strictly forbid direct transition from CONFIRMED_PRESENT to CONFIRMED_REMOVED on initial non-detection.",
            "B2": "engine.py:983 unconditionally halts absence reasoning on registration failure or missing overlap mask. Occlusion IoU check (0.20) is hardcoded in engine.py:1031.",
            "A1": "engine.py:83 returns False when registration.success is False regardless of inlier ratio. engine.py:985 also halts absence if overlap_mask is None. No engine bypass flag exists.",
            "A4": "engine.py:1070 hardcodes target_state = PENDING_REMOVAL for CONFIRMED_PRESENT entities on visit 1. vacancy_visits_for_removal=1 only affects subsequent visits.",
            "A6": "engine.py:346 loads all entities (including CONFIRMED_REMOVED) and engine.py:614 re-identifies them. No configuration flag exists to filter removed entities.",
        }
        return reasons.get(cid, "Configuration is blocked in the current engine implementation.")

    @staticmethod
    def _get_partial_reason(cid: str) -> str:
        reasons = {
            "A2": "vacancy_similarity_max=1.0 forces patch_sim < 1.0 to evaluate as vacant for valid crops. However, missing aligned images or degenerate crops still fall back to unobserved streak logic.",
            "A3": "min_overlap_fraction=0.0 disables bounding box FOV clipping. However, inter-detection occlusion IoU (0.20) in engine.py:1031 remains active because it is hardcoded.",
        }
        return reasons.get(cid, "Configuration is partially supported with caveats.")
