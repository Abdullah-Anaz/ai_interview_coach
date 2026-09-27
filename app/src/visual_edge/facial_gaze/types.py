from dataclasses import dataclass


@dataclass(frozen=True)
class FacialDescriptorPayload:
    """
    An immutable transport structure containing the regressed OCEAN psychometric 
    scores and the finalized BeMERC translated text for a specific temporal segment.
    """
    segment_id: str
    start_time_sec: float
    end_time_sec: float
    behavioral_descriptor: str
    extraversion_score: float
    neuroticism_score: float
    agreeableness_score: float
    conscientiousness_score: float
    openness_score: float

    def __post_init__(self) -> None:
        """
        Executes strict runtime validation to ensure mathematical and structural integrity.

        Raises:
            TypeError: If attribute types deviate from the defined schema.
            ValueError: If temporal boundaries are inverted, negative, or identifiers are empty.
        """
        if not isinstance(self.segment_id, str):
            raise TypeError("segment_id must be a string.")
        if not self.segment_id.strip():
            raise ValueError("segment_id cannot be an empty string.")

        if not isinstance(self.start_time_sec, (int, float)):
            raise TypeError("start_time_sec must be a numeric value.")
        if not isinstance(self.end_time_sec, (int, float)):
            raise TypeError("end_time_sec must be a numeric value.")
            
        if self.start_time_sec < 0.0:
            raise ValueError("start_time_sec cannot be a negative value.")
        if self.end_time_sec <= self.start_time_sec:
            raise ValueError("end_time_sec must be strictly greater than start_time_sec.")

        if not isinstance(self.behavioral_descriptor, str):
            raise TypeError("behavioral_descriptor must be a string.")
            
        for score_name in [
            "extraversion_score", "neuroticism_score", "agreeableness_score", 
            "conscientiousness_score", "openness_score"
        ]:
            val = getattr(self, score_name)
            if not isinstance(val, (int, float)):
                raise TypeError(f"{score_name} must be a numeric value.")