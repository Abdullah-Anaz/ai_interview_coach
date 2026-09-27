from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticDescriptorPayload:
    """
    An immutable transport structure containing the extracted verbatim 
    raw transcript for a specific temporally aligned segment.

    Attributes:
        segment_id (int): Sequence identifier mapped to the source utterance.
        start_time_sec (float): Absolute lower temporal boundary in seconds.
        end_time_sec (float): Absolute upper temporal boundary in seconds.
        transcript_text (str): The concatenated, whitespace-normalized spoken text.
    """
    segment_id: int
    start_time_sec: float
    end_time_sec: float
    transcript_text: str

    def __post_init__(self) -> None:
        """
        Executes strict runtime validation to ensure structural and temporal integrity.

        Raises:
            TypeError: If attribute types deviate from the defined schema.
            ValueError: If temporal boundaries are inverted, negative, or identifiers are invalid.
        """
        if not isinstance(self.segment_id, int):
            raise TypeError("segment_id must be an integer.")
        if self.segment_id < 0:
            raise ValueError("segment_id cannot be a negative integer.")

        if not isinstance(self.start_time_sec, (int, float)):
            raise TypeError("start_time_sec must be a numeric value.")
        if not isinstance(self.end_time_sec, (int, float)):
            raise TypeError("end_time_sec must be a numeric value.")
            
        if self.start_time_sec < 0.0:
            raise ValueError("start_time_sec cannot be a negative value.")
        if self.end_time_sec <= self.start_time_sec:
            raise ValueError("end_time_sec must be strictly greater than start_time_sec.")

        if not isinstance(self.transcript_text, str):
            raise TypeError("transcript_text must be a string.")