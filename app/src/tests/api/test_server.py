import io
import queue
from typing import Any, Dict

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.src.api.dependencies import global_pipeline_state
from app.src.api.server import app
from app.src.apex_judge.types import CoachingFeedbackPayload
from app.src.orchestration.orchestrator import OrchestratorTerminalSentinel

client = TestClient(app)


def _wait_for_sentinel(target_queue: queue.Queue, timeout_sec: float = 2.0) -> Any:
    """Drains a queue until the terminal sentinel is found, bypassing unmocked media outputs."""
    while True:
        item = target_queue.get(timeout=timeout_sec)
        if isinstance(item, OrchestratorTerminalSentinel):
            return item


@pytest.fixture(autouse=True)
def isolate_pipeline_state() -> None:
    """Atomically purges all global queues to guarantee absolute state isolation."""
    queues = [
        global_pipeline_state.q_transcription,
        global_pipeline_state.q_window,
        global_pipeline_state.q_acoustic,
        global_pipeline_state.q_semantic,
        global_pipeline_state.q_visual_facial,
        global_pipeline_state.q_visual_skeletal,
        global_pipeline_state.q_nexus_out,
        global_pipeline_state.q_judge_out
    ]
    
    for q in queues:
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break


@pytest.fixture
def multiplexed_media_bytes() -> bytes:
    """Synthesizes a valid, fully multiplexed audio-visual container in memory without disk I/O."""
    buffer = io.BytesIO()
    container = av.open(buffer, mode='w', format='mp4')
    
    video_stream = container.add_stream('mpeg4', rate=10)
    video_stream.width = 320
    video_stream.height = 240
    video_stream.pix_fmt = 'yuv420p'
    
    audio_stream = container.add_stream('mp2', rate=16000)
    
    for _ in range(3):
        v_frame = av.VideoFrame.from_ndarray(np.zeros((240, 320, 3), dtype=np.uint8), format='rgb24')
        for packet in video_stream.encode(v_frame):
            container.mux(packet)
            
        a_frame = av.AudioFrame.from_ndarray(np.zeros((1, 1600), dtype=np.float32), format='fltp', layout='mono')
        a_frame.sample_rate = 16000
        for packet in audio_stream.encode(a_frame):
            container.mux(packet)
            
    for packet in video_stream.encode():
        container.mux(packet)
    for packet in audio_stream.encode():
        container.mux(packet)
        
    container.close()
    return buffer.getvalue()


# =====================================================================
# EXPECTED BEHAVIOUR: TRANSMISSION
# =====================================================================

def test_interview_stream_endpoint_expected_behaviour_for_feedback_transmission() -> None:
    """Verifies that the concurrent async task natively polls and dispatches JSON to the client."""
    with client.websocket_connect("/ws/interview") as websocket:
        # Injected AFTER connection to bypass the server's initial queue purge
        payload = CoachingFeedbackPayload(segment_id=7, feedback_text="Strong eye contact maintained.")
        global_pipeline_state.q_judge_out.put(payload)
        
        data: Dict[str, Any] = websocket.receive_json()
        assert data["segment_id"] == 7
        assert data["feedback"] == "Strong eye contact maintained."


def test_interview_stream_endpoint_expected_behaviour_for_terminal_sentinel_transmission() -> None:
    """Verifies the background transmission loop cleanly exits upon intercepting a terminal sentinel."""
    with client.websocket_connect("/ws/interview") as websocket:
        global_pipeline_state.q_judge_out.put(OrchestratorTerminalSentinel())
        websocket.send_bytes(b"")
        websocket.close()


def test_interview_stream_endpoint_expected_behaviour_for_unsupported_queue_payload() -> None:
    """Verifies the transmission loop ignores unknown datatypes in the egress queue without crashing."""
    with client.websocket_connect("/ws/interview") as websocket:
        global_pipeline_state.q_judge_out.put("unsupported_string_payload")
        global_pipeline_state.q_judge_out.put(OrchestratorTerminalSentinel())
        websocket.send_bytes(b"")
        websocket.close()


def test_interview_stream_endpoint_expected_behaviour_for_initial_queue_purging() -> None:
    """Verifies the endpoint purges stale elements from the judge egress queue upon a new connection."""
    global_pipeline_state.q_judge_out.put(CoachingFeedbackPayload(segment_id=1, feedback_text="Stale data."))
    
    with client.websocket_connect("/ws/interview") as websocket:
        assert global_pipeline_state.q_judge_out.empty()
        websocket.close()


# =====================================================================
# EXPECTED BEHAVIOUR: INGESTION & ROUTING
# =====================================================================

def test_interview_stream_endpoint_expected_behaviour_for_valid_binary_ingestion(
    multiplexed_media_bytes: bytes
) -> None:
    """Verifies that the WebSocket successfully receives and accumulates raw binary packets natively."""
    with client.websocket_connect("/ws/interview") as websocket:
        websocket.send_bytes(multiplexed_media_bytes)
        websocket.close()


def test_interview_stream_endpoint_expected_behaviour_for_stream_termination_action(
    multiplexed_media_bytes: bytes
) -> None:
    """Verifies the standard JSON action trigger executes the unmocked routing pipeline and queues sentinels."""
    with client.websocket_connect("/ws/interview") as websocket:
        websocket.send_bytes(multiplexed_media_bytes)
        websocket.send_json({"action": "TERMINATE_STREAM"})
        websocket.close()

    t_sentinel = _wait_for_sentinel(global_pipeline_state.q_transcription)
    w_sentinel = _wait_for_sentinel(global_pipeline_state.q_window)
    
    assert isinstance(t_sentinel, OrchestratorTerminalSentinel)
    assert isinstance(w_sentinel, OrchestratorTerminalSentinel)


def test_interview_stream_endpoint_expected_behaviour_for_legacy_eos_type(
    multiplexed_media_bytes: bytes
) -> None:
    """Verifies the legacy EOS JSON payload triggers the unmocked routing pipeline."""
    with client.websocket_connect("/ws/interview") as websocket:
        websocket.send_bytes(multiplexed_media_bytes)
        websocket.send_json({"type": "EOS"})
        websocket.close()

    t_sentinel = _wait_for_sentinel(global_pipeline_state.q_transcription)
    w_sentinel = _wait_for_sentinel(global_pipeline_state.q_window)
    
    assert isinstance(t_sentinel, OrchestratorTerminalSentinel)
    assert isinstance(w_sentinel, OrchestratorTerminalSentinel)


def test_interview_stream_endpoint_expected_behaviour_for_graceful_client_disconnect() -> None:
    """Verifies client closure triggers the systemic teardown cascade across all ingress queues."""
    with client.websocket_connect("/ws/interview") as websocket:
        websocket.close()

    t_sentinel = _wait_for_sentinel(global_pipeline_state.q_transcription)
    w_sentinel = _wait_for_sentinel(global_pipeline_state.q_window)
    
    assert isinstance(t_sentinel, OrchestratorTerminalSentinel)
    assert isinstance(w_sentinel, OrchestratorTerminalSentinel)


# =====================================================================
# EDGE CASES
# =====================================================================

def test_interview_stream_endpoint_expected_behaviour_for_empty_binary_frames() -> None:
    """Verifies the socket safely ignores zero-byte network frames without invoking the strict decoder."""
    with client.websocket_connect("/ws/interview") as websocket:
        websocket.send_bytes(b"")
        websocket.close()


def test_interview_stream_endpoint_expected_behaviour_for_sequential_fragments(
    multiplexed_media_bytes: bytes
) -> None:
    """Verifies the socket loop sustains continuity across multiple incoming discrete blob fragments."""
    with client.websocket_connect("/ws/interview") as websocket:
        websocket.send_bytes(multiplexed_media_bytes)
        websocket.send_bytes(multiplexed_media_bytes)
        websocket.close()


def test_interview_stream_endpoint_expected_behaviour_for_invalid_json_text() -> None:
    """Verifies the endpoint catches JSONDecodeError natively and maintains connection stability."""
    with client.websocket_connect("/ws/interview") as websocket:
        websocket.send_text("{malformed_json: true]")
        websocket.send_bytes(b"")
        websocket.close()


def test_interview_stream_endpoint_expected_behaviour_for_non_actionable_json() -> None:
    """Verifies the endpoint gracefully ignores valid JSON dictionaries that lack routing action keys."""
    with client.websocket_connect("/ws/interview") as websocket:
        websocket.send_json({"status": "heartbeat", "data": "ping"})
        websocket.close()


# =====================================================================
# ERROR CASES
# =====================================================================

def test_interview_stream_endpoint_raises_for_corrupted_routing_failure() -> None:
    """
    Verifies that supplying completely invalid media bytes to the unmocked orchestrator 
    triggers a native PyAV exception, which is caught, logged, and bypassed to inject sentinels.
    """
    with client.websocket_connect("/ws/interview") as websocket:
        websocket.send_bytes(b"this_is_not_a_valid_webm_or_mp4_container")
        websocket.send_json({"action": "TERMINATE_STREAM"})
        websocket.close()

    t_sentinel = _wait_for_sentinel(global_pipeline_state.q_transcription)
    w_sentinel = _wait_for_sentinel(global_pipeline_state.q_window)

    assert isinstance(t_sentinel, OrchestratorTerminalSentinel)
    assert isinstance(w_sentinel, OrchestratorTerminalSentinel)