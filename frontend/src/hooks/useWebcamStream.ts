import { useState, useEffect, useCallback, useRef } from "react";

export interface WebcamStreamState {
  /** The active MediaStream instance bound to the user's hardware. */
  stream: MediaStream | null;
  /** Represents if all audio tracks are currently disabled. */
  isAudioMuted: boolean;
  /** Represents if all video tracks are currently disabled. */
  isVideoDisabled: boolean;
  /** Captured hardware or permission errors during initialization. */
  error: Error | null;
  /** Toggles the enabled state of all audio tracks in the active stream. */
  toggleAudio: () => void;
  /** Toggles the enabled state of all video tracks in the active stream. */
  toggleVideo: () => void;
  /** Immediately terminates all active hardware tracks and purges the stream. */
  stopStream: () => void;
}

/**
 * Initializes and manages a secure WebRTC local media stream, encapsulating hardware
 * permissions, track states, and lifecycle teardown.
 *
 * @returns {WebcamStreamState} A strictly typed state object exposing the active stream and its controls.
 */
export function useWebcamStream(): WebcamStreamState {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isAudioMuted, setIsAudioMuted] = useState<boolean>(false);
  const [isVideoDisabled, setIsVideoDisabled] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    let active = true;

    const initializeMedia = async (): Promise<void> => {
      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error("MediaDevices API is unsupported in this environment. HTTPS may be required.");
        }

        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user",
          },
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });

        if (active) {
          setStream(mediaStream);
          streamRef.current = mediaStream;
        } else {
          mediaStream.getTracks().forEach((track) => track.stop());
        }
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err : new Error("Failed to initialize hardware stream."));
        }
      }
    };

    void initializeMedia();

    return () => {
      active = false;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, []);

  const toggleAudio = useCallback((): void => {
    if (!streamRef.current) return;
    const audioTracks = streamRef.current.getAudioTracks();
    audioTracks.forEach((track) => {
      track.enabled = !track.enabled;
    });
    setIsAudioMuted((prev) => !prev);
  }, []);

  const toggleVideo = useCallback((): void => {
    if (!streamRef.current) return;
    const videoTracks = streamRef.current.getVideoTracks();
    videoTracks.forEach((track) => {
      track.enabled = !track.enabled;
    });
    setIsVideoDisabled((prev) => !prev);
  }, []);

  const stopStream = useCallback((): void => {
    if (!streamRef.current) return;
    streamRef.current.getTracks().forEach((track) => track.stop());
    setStream(null);
    streamRef.current = null;
  }, []);

  return {
    stream,
    isAudioMuted,
    isVideoDisabled,
    error,
    toggleAudio,
    toggleVideo,
    stopStream,
  };
}