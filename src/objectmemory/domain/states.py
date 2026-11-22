"""
State definitions and epistemic taxonomy for ObjectMemory.

Safeguard:
Separates inferred persistent environmental states from observation visibility
and pairwise baseline prediction types.
"""

from enum import Enum
from typing import Dict, Set


class EpistemicState(str, Enum):
    """
    Inferred real-world status of a persistent physical object on the workbench.

    Safeguard:
    Missing observations do NOT imply CONFIRMED_REMOVED. Lack of evidence
    must be represented as INSUFFICIENT_EVIDENCE or PENDING_REMOVAL.
    """
    CONFIRMED_PRESENT = "CONFIRMED_PRESENT"
    CONFIRMED_MOVED = "CONFIRMED_MOVED"
    CONFIRMED_REMOVED = "CONFIRMED_REMOVED"
    CONFIRMED_ADDED = "CONFIRMED_ADDED"
    PENDING_REMOVAL = "PENDING_REMOVAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AMBIGUOUS_IDENTITY = "AMBIGUOUS_IDENTITY"


class VisibilityStatus(str, Enum):
    """
    Observational visibility of an entity's canonical region in a specific visit image.
    Decoupled from detector confidence and entity persistent state.
    """
    CLEARLY_VISIBLE = "CLEARLY_VISIBLE"
    PARTIALLY_OCCLUDED = "PARTIALLY_OCCLUDED"
    HEAVILY_OCCLUDED = "HEAVILY_OCCLUDED"
    OUT_OF_FRAME = "OUT_OF_FRAME"
    UNOBSERVED = "UNOBSERVED"
    UNCERTAIN = "UNCERTAIN"


class BaselineChangeType(str, Enum):
    """
    Pairwise change detection prediction types, kept isolated from temporal memory.
    """
    UNCHANGED = "UNCHANGED"
    MOVED = "MOVED"
    ADDED = "ADDED"
    REMOVED = "REMOVED"
    UNKNOWN = "UNKNOWN"


# Explicit, small, documented state transition policy
VALID_STATE_TRANSITIONS: Dict[EpistemicState, Set[EpistemicState]] = {
    EpistemicState.INSUFFICIENT_EVIDENCE: {
        EpistemicState.INSUFFICIENT_EVIDENCE,
        EpistemicState.CONFIRMED_PRESENT,
        EpistemicState.CONFIRMED_MOVED,
        EpistemicState.PENDING_REMOVAL,
        EpistemicState.AMBIGUOUS_IDENTITY,
    },
    EpistemicState.CONFIRMED_ADDED: {
        EpistemicState.CONFIRMED_ADDED,
        EpistemicState.CONFIRMED_PRESENT,
        EpistemicState.CONFIRMED_MOVED,
        EpistemicState.PENDING_REMOVAL,
        EpistemicState.INSUFFICIENT_EVIDENCE,
        EpistemicState.AMBIGUOUS_IDENTITY,
    },
    EpistemicState.CONFIRMED_PRESENT: {
        EpistemicState.CONFIRMED_PRESENT,
        EpistemicState.CONFIRMED_MOVED,
        EpistemicState.PENDING_REMOVAL,
        EpistemicState.INSUFFICIENT_EVIDENCE,
        EpistemicState.AMBIGUOUS_IDENTITY,
        # Direct transition to CONFIRMED_REMOVED is rejected.
        # Policy mandates passing through PENDING_REMOVAL to prevent hasty absence declaration.
    },
    EpistemicState.CONFIRMED_MOVED: {
        EpistemicState.CONFIRMED_MOVED,
        EpistemicState.CONFIRMED_PRESENT,
        EpistemicState.PENDING_REMOVAL,
        EpistemicState.INSUFFICIENT_EVIDENCE,
        EpistemicState.AMBIGUOUS_IDENTITY,
    },
    EpistemicState.PENDING_REMOVAL: {
        EpistemicState.PENDING_REMOVAL,
        EpistemicState.CONFIRMED_REMOVED,  # Verified removal after pending confirmation
        EpistemicState.CONFIRMED_PRESENT,  # Re-detected at original spot
        EpistemicState.CONFIRMED_MOVED,    # Found relocated elsewhere
        EpistemicState.INSUFFICIENT_EVIDENCE,
        EpistemicState.AMBIGUOUS_IDENTITY,
    },
    EpistemicState.CONFIRMED_REMOVED: {
        EpistemicState.CONFIRMED_REMOVED,
        EpistemicState.CONFIRMED_PRESENT,  # Object re-introduced to workbench
        EpistemicState.CONFIRMED_ADDED,
        EpistemicState.INSUFFICIENT_EVIDENCE,
    },
    EpistemicState.AMBIGUOUS_IDENTITY: {
        EpistemicState.AMBIGUOUS_IDENTITY,
        EpistemicState.CONFIRMED_PRESENT,
        EpistemicState.CONFIRMED_MOVED,
        EpistemicState.PENDING_REMOVAL,
        EpistemicState.INSUFFICIENT_EVIDENCE,
    },
}


def is_valid_transition(prior_state: EpistemicState, new_state: EpistemicState) -> bool:
    """Returns True if the transition from prior_state to new_state is allowed by policy."""
    allowed = VALID_STATE_TRANSITIONS.get(prior_state, set())
    return new_state in allowed
