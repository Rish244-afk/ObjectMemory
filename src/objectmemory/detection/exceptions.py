"""
Exceptions for object detection and model management in ObjectMemory.
"""

from __future__ import annotations


class DetectorError(Exception):
    """Base exception for all detector-related failures."""
    pass


class DetectorUnavailableError(DetectorError, FileNotFoundError):
    """
    Raised when a requested detector is unavailable due to missing weights,
    missing dependencies, or unsupported hardware/architecture.
    Subclasses FileNotFoundError for backward compatibility.
    """
    pass


class DetectorConfigurationError(DetectorError, ValueError):
    """
    Raised when invalid detector configuration parameters are supplied.
    Subclasses ValueError for backward compatibility.
    """
    pass


class DetectorInferenceError(DetectorError, RuntimeError):
    """
    Raised when a detector encounters an unrecoverable failure during forward inference.
    Subclasses RuntimeError for backward compatibility.
    """
    pass
