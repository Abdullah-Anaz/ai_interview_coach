import { useState, useRef, useCallback, useEffect } from "react";
import type {
  ConnectionState,
  CoachingFeedbackPayload,
  PipelineModalityStatus,
  BackendIncomingMessage,
} from "../types/backend";

export interface GcpSocketState {
  /** The current lifecycle state of the WebSocket connection and recording session. */
  connectionState: ConnectionState;
  /** Real-time modality processing statuses emitted by the Fusion Nexus. */
  modalityStatus: PipelineModalityStatus | null;
  /** The final markdown evaluation payload received from the Gemini judge. */
  evaluation: CoachingFeedbackPayload | null;
  /** Captured string detailing any network or pipeline errors. */
  error: string | null;
  /** Initializes the WebSocket connection to the GCP VM. */
  connect: () => void;
  /** Binds the MediaRecorder to the active stream and begins transmitting binary chunks. */
  startRecording: () => void;
  /** Halts transmission, emits an End-Of-Stream sentinel, and awaits final evaluation. */
  stopRecording: () => void;
  /** Forcefully terminates the socket connection and cleans up resources. */
  disconnect: () => void;
}

/**
 * Manages the WebSocket telemetry link to the GCP backend, handling binary media streaming,
 * JSON state tracking, and connection lifecycle management.
 *
 * @param {MediaStream | null} stream - The active media stream from the local hardware.
 * @returns {GcpSocketState} Exposes the reactive state and network control methods.
 */
export function useGcpSocket(stream: MediaStream | null): GcpSocketState {
  const [connectionState, setConnectionState] = useState<ConnectionState>("IDLE");
  const [modalityStatus, setModalityStatus] = useState<PipelineModalityStatus | null>(null);
  const [evaluation, setEvaluation] = useState<CoachingFeedbackPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  const disconnect = useCallback((): void => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
    mediaRecorderRef.current = null;
    setConnectionState("IDLE");
  }, []);

  const connect = useCallback((): void => {
    const wsUrl = process.env.NEXT_PUBLIC_GCP_WS_URL;
    if (!wsUrl) {
      setError("GCP WebSocket URL is not configured in the environment.");
      setConnectionState("ERROR");
      return;
    }

    // Prevent duplicate connection attempts if already connecting or open
    if (socketRef.current?.readyState === WebSocket.OPEN || socketRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    setConnectionState("CONNECTING");
    setError(null);

    const ws = new WebSocket(wsUrl);
    socketRef.current = ws; // Assign immediately to track instance

    ws.onopen = () => {
      // Ensure this event belongs to the currently active socket instance
      if (socketRef.current === ws) {
        setConnectionState("CONNECTED");
      }
    };

    ws.onmessage = (event: MessageEvent) => {
      if (socketRef.current !== ws) return;
      try {
        const message = JSON.parse(event.data as string);

        // 1. Check for the new flat JSON structure from the optimized backend
        if (message.feedback !== undefined && message.segment_id !== undefined) {
          
          // (Optional Cleanup) The Gemini API in your backend logs showed it is returning a stringified 
          // Python list: "[{'type': 'text', 'text': '...'}]". This safely extracts just the markdown text.
          let cleanFeedback = message.feedback;
          if (typeof cleanFeedback === 'string' && cleanFeedback.includes("'text': '")) {
             try {
                 const match = cleanFeedback.match(/'text':\s*'([\s\S]*?)'(?:,|})/);
                 if (match && match[1]) cleanFeedback = match[1].replace(/\\n/g, '\n');
             } catch (e) {
                 console.warn("Failed to parse Gemini Python string", e);
             }
          }

          setEvaluation({
            segment_id: Number(message.segment_id),
            feedback_text: cleanFeedback,
          });
          
          setConnectionState("EVALUATION_READY");
          
          // CRITICAL FIX: Do NOT call disconnect() here! 
          // disconnect() hardcodes setConnectionState("IDLE"), which instantly overwrites EVALUATION_READY.
          // Instead, close the socket directly so the ws.onclose handler can safely preserve your state.
          if (socketRef.current) {
            socketRef.current.close();
          }
          return;
        }

        // 2. Legacy fallback for old message wrappers (if any)
        switch (message.type) {
          case "modality_status":
            setModalityStatus(message.payload);
            break;
          case "terminal_error":
            setError(message.error);
            setConnectionState("ERROR");
            disconnect();
            break;
        }
      } catch (parseError) {
        console.error("Failed to parse incoming GCP telemetry:", parseError);
      }
    };

    ws.onerror = () => {
      if (socketRef.current === ws) {
        setError("WebSocket connection encountered a network error.");
        setConnectionState("ERROR");
      }
    };

    ws.onclose = () => {
      if (socketRef.current === ws) {
        setConnectionState((prev) => (prev === "EVALUATION_READY" ? prev : "IDLE"));
        socketRef.current = null;
      }
    };
  }, [disconnect]);

 const startRecording = useCallback((): void => {
    if (!stream) {
      console.error("[DEBUG] No active hardware stream provided to startRecording.");
      setError("No active hardware stream available for transmission.");
      return;
    }

    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      console.error("[DEBUG] Socket is not open. State:", socketRef.current?.readyState);
      setError("Cannot start recording: GCP connection is not established.");
      return;
    }

    try {
      const mediaRecorder = new MediaRecorder(stream);
      console.log("[DEBUG] MediaRecorder created successfully. Initial state:", mediaRecorder.state);

      mediaRecorder.ondataavailable = (event: BlobEvent) => {
        console.log("[DEBUG] ondataavailable fired. Blob size:", event.data?.size, "bytes");
        if (event.data && event.data.size > 0 && socketRef.current?.readyState === WebSocket.OPEN) {
          socketRef.current.send(event.data);
        } else {
          console.warn("[DEBUG] Dropped chunk! Size was 0 or socket was closed.");
        }
      };

      mediaRecorder.onerror = (event: Event) => {
        console.error("[DEBUG] MediaRecorder error encountered:", event);
      };

      mediaRecorder.start(1000);
      console.log("[DEBUG] MediaRecorder started with timeslice: 1000ms. Current state:", mediaRecorder.state);
      
      mediaRecorderRef.current = mediaRecorder;
      setConnectionState("RECORDING");
    } catch (err) {
      console.error("[DEBUG] Exception thrown while starting MediaRecorder:", err);
      setError(err instanceof Error ? err.message : "Failed to bind MediaRecorder to stream.");
      setConnectionState("ERROR");
    }
  }, [stream]);

  const stopRecording = useCallback((): void => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.requestData();
      mediaRecorderRef.current.stop();
    }

    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      const eosSentinel = JSON.stringify({ type: "EOS" });
      socketRef.current.send(eosSentinel);
      setConnectionState("PROCESSING");
      // CRITICAL: Do NOT close or disconnect here! Leave the socket open 
      // so the backend can transmit the final AI evaluation back to the client.
    } else {
      setConnectionState("PROCESSING");
    }
  }, []);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    connectionState,
    modalityStatus,
    evaluation,
    error,
    connect,
    startRecording,
    stopRecording,
    disconnect,
  };
}