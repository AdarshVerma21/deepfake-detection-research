"""Custom error types for robust pipeline failures."""


class DeepfakeResearchError(Exception):
    """Base exception for project-specific errors."""


class ConfigurationError(DeepfakeResearchError):
    """Raised when configuration loading or validation fails."""


class DeviceConfigurationError(DeepfakeResearchError):
    """Raised when runtime device settings are invalid."""
