"""Custom exceptions for CCTV detection application."""


class CCTVDetectionError(Exception):
    """Base exception for all CCTV detection errors."""

    pass


class ValidationError(CCTVDetectionError):
    """Raised when input validation fails."""

    pass


class DatasetPreparationError(CCTVDetectionError):
    """Raised when dataset preparation fails."""

    pass


class TrainingError(CCTVDetectionError):
    """Raised when model training fails."""

    pass


class InferenceError(CCTVDetectionError):
    """Raised when model inference fails."""

    pass


class ConfigurationError(CCTVDetectionError):
    """Raised when configuration is invalid."""

    pass
