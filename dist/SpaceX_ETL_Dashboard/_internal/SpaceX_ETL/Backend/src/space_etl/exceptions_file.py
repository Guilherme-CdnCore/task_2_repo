
class SpaceETLError(Exception):
    """Base exception for SpaceX ETL errors."""

    pass


class DataExtractionError(SpaceETLError):
    """Raised when data extraction fails."""
    pass


class DataTransformationError(SpaceETLError):
    """Raised when data transformation fails."""
    pass


class DataLoadError(SpaceETLError):
    """Raised when data loading or writing fails."""
    pass


class InvalidDatetimeError(SpaceETLError):
    """Raised when datetime parsing fails."""
    pass


class HashComputationError(SpaceETLError):
    """Raised when hash computation fails."""
    pass


class ConfigError(SpaceETLError):
    """Raised when there is a configuration issue."""
    pass


class FileNotFoundError(SpaceETLError):
    """Raised when a required file is missing."""
    pass


class SafeGetError(SpaceETLError):
    """Raised when safe_get fails to retrieve a value."""
    pass