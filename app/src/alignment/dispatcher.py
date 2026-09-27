from queue import Full, Queue

from app.src.orchestration.types import SynchronizedSegment


def dispatch_synchronized_segment(
    segment: SynchronizedSegment,
    acoustic_queue: Queue,
    visual_queue: Queue,
    semantic_queue: Queue,
    timeout_sec: float = 5.0
) -> None:
    """
    Distributes a unified multimodal segment to all downstream analysis branches.

    Args:
        segment (SynchronizedSegment): The temporal-locked payload to distribute.
        acoustic_queue (Queue): Ingress queue for the Acoustic Edge (vocal descriptors).
        visual_queue (Queue): Ingress queue for the Visual Edge (vision kinematics).
        semantic_queue (Queue): Ingress queue for the Semantic Edge (raw transcript).
        timeout_sec (float): Maximum block time before aborting due to downstream saturation.

    Raises:
        RuntimeError: If any downstream queue is saturated and blocks beyond the timeout.
    """
    try:
        acoustic_queue.put(segment, timeout=timeout_sec)
    except Full as e:
        raise RuntimeError("Acoustic Edge queue saturation prevented segment dispatch.") from e

    try:
        visual_queue.put(segment, timeout=timeout_sec)
    except Full as e:
        raise RuntimeError("Visual Edge queue saturation prevented segment dispatch.") from e

    try:
        semantic_queue.put(segment, timeout=timeout_sec)
    except Full as e:
        raise RuntimeError("Semantic Edge queue saturation prevented segment dispatch.") from e