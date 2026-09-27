from dataclasses import dataclass


@dataclass(frozen=True)
class SkeletalDescriptorPayload:
    """
    Immutable data transfer object transporting mathematically derived behavioral
    descriptors from the Human-Omni temporal pipeline to the Late Fusion Nexus.

    Attributes:
        segment_id (int): A unique chronological identifier for the video sequence.
        start_time_sec (float): The temporal start boundary of the rolling window.
        end_time_sec (float): The temporal end boundary of the rolling window.
        behavioral_descriptor (str): The discrete BeMERC natural language translation.
        perceptual_jitter (float): The raw, high-frequency spatial variance metric.
        temporal_smoothness (float): The raw, low-frequency spatial trajectory metric.
    """
    segment_id: int
    start_time_sec: float
    end_time_sec: float
    behavioral_descriptor: str
    perceptual_jitter: float
    temporal_smoothness: float

    def __post_init__(self) -> None:
        """
        Enforces strict runtime type and boundary validation.

        Raises:
            TypeError: If any attribute fails strict type constraints.
            ValueError: If temporal boundaries are logically invalid or text is empty.
        """
        if not isinstance(self.segment_id, int) or isinstance(self.segment_id, bool):
            raise TypeError("segment_id must be an integer.")
        if self.segment_id < 0:
            raise ValueError("segment_id must be non-negative.")

        if not isinstance(self.start_time_sec, (int, float)) or isinstance(self.start_time_sec, bool):
            raise TypeError("start_time_sec must be a numeric float.")
        if self.start_time_sec < 0.0:
            raise ValueError("start_time_sec cannot be negative.")

        if not isinstance(self.end_time_sec, (int, float)) or isinstance(self.end_time_sec, bool):
            raise TypeError("end_time_sec must be a numeric float.")
        if self.end_time_sec <= self.start_time_sec:
            raise ValueError("end_time_sec must be strictly greater than start_time_sec.")

        if not isinstance(self.behavioral_descriptor, str):
            raise TypeError("behavioral_descriptor must be a string.")
        if not self.behavioral_descriptor.strip():
            raise ValueError("behavioral_descriptor cannot be empty or whitespace.")

        if not isinstance(self.perceptual_jitter, (int, float)) or isinstance(self.perceptual_jitter, bool):
            raise TypeError("perceptual_jitter must be a numeric float.")

        if not isinstance(self.temporal_smoothness, (int, float)) or isinstance(self.temporal_smoothness, bool):
            raise TypeError("temporal_smoothness must be a numeric float.")