"use client";

import { useCallback, useState } from "react";
import type { ReactElement } from "react";
import { Loader2, Mic, MicOff, Square, Video, VideoOff } from "lucide-react";

export interface RecordingControlsProps {
  /** Indicates if the application is currently actively streaming binary chunks. */
  isRecording: boolean;
  /** Indicates if the application is currently waiting for the GCP backend evaluation. */
  isProcessing: boolean;
  /** Current hardware state indicating if audio tracks are disabled. */
  isAudioMuted: boolean;
  /** Current hardware state indicating if video tracks are disabled. */
  isVideoDisabled: boolean;
  /** Callback to invert the current audio track enablement state. */
  onToggleAudio: () => void;
  /** Callback to invert the current video track enablement state. */
  onToggleVideo: () => void;
  /** Callback triggered to halt the recording and request the final evaluation. */
  onEndAnswer: () => Promise<void> | void;
  /** Optional Tailwind class string for external layout positioning. */
  className?: string;
}

/**
 * Renders the primary user interaction surface for hardware toggles and telemetry transmission control.
 *
 * @param {RecordingControlsProps} props - The operational state and hardware callbacks.
 * @returns {ReactElement} A high-tech, glassmorphism control strip.
 */
export function RecordingControls({
  isRecording,
  isProcessing,
  isAudioMuted,
  isVideoDisabled,
  onToggleAudio,
  onToggleVideo,
  onEndAnswer,
  className = "",
}: RecordingControlsProps): ReactElement {
  const [isHandlingStop, setIsHandlingStop] = useState<boolean>(false);

  const handleEndAction = useCallback(async (): Promise<void> => {
    if (isHandlingStop || isProcessing || !isRecording) {
      return;
    }

    try {
      setIsHandlingStop(true);
      await Promise.resolve(onEndAnswer());
    } catch (error) {
      console.error("Failed to execute answer termination sequence:", error);
    } finally {
      setIsHandlingStop(false);
    }
  }, [isHandlingStop, isProcessing, isRecording, onEndAnswer]);

  return (
    <div
      className={`flex items-center justify-center gap-6 rounded-2xl border border-white/10 bg-black/60 p-4 shadow-[0_0_50px_rgba(0,0,0,0.8)] backdrop-blur-xl ring-1 ring-white/5 ${className}`}
    >
      <button
        onClick={onToggleAudio}
        disabled={isProcessing}
        type="button"
        className={`group relative flex h-12 w-12 items-center justify-center rounded-full border transition-all duration-300 disabled:opacity-50 ${
          isAudioMuted
            ? "border-red-500/30 bg-red-500/10 text-red-400 hover:bg-red-500/20 hover:shadow-[0_0_15px_rgba(239,68,68,0.2)]"
            : "border-white/10 bg-white/5 text-neutral-300 hover:bg-white/10 hover:text-white hover:shadow-[0_0_15px_rgba(255,255,255,0.1)]"
        }`}
        aria-label={isAudioMuted ? "Unmute Microphone" : "Mute Microphone"}
      >
        {isAudioMuted ? (
          <MicOff className="h-5 w-5 transition-transform group-hover:scale-110" strokeWidth={1.5} />
        ) : (
          <Mic className="h-5 w-5 transition-transform group-hover:scale-110" strokeWidth={1.5} />
        )}
      </button>

      <button
        onClick={handleEndAction}
        disabled={!isRecording || isProcessing || isHandlingStop}
        type="button"
        className={`relative flex h-14 w-48 items-center justify-center gap-3 rounded-full border border-red-500/50 bg-red-500/10 transition-all duration-500 disabled:cursor-not-allowed disabled:border-white/5 disabled:bg-neutral-900/50 ${
          isRecording && !isProcessing && !isHandlingStop
            ? "hover:border-red-400 hover:bg-red-500/20 hover:shadow-[0_0_30px_rgba(239,68,68,0.3)]"
            : ""
        }`}
        aria-label="End Answer"
      >
        {isProcessing || isHandlingStop ? (
          <>
            <Loader2 className="h-5 w-5 animate-spin text-cyan-400" strokeWidth={2} />
            <span className="font-mono text-sm font-medium uppercase tracking-widest text-cyan-400">
              Processing
            </span>
          </>
        ) : (
          <>
            <Square className="h-4 w-4 fill-red-500 text-red-500" strokeWidth={2} />
            <span className="font-mono text-sm font-semibold uppercase tracking-widest text-red-50">
              End Answer
            </span>
          </>
        )}
      </button>

      <button
        onClick={onToggleVideo}
        disabled={isProcessing}
        type="button"
        className={`group relative flex h-12 w-12 items-center justify-center rounded-full border transition-all duration-300 disabled:opacity-50 ${
          isVideoDisabled
            ? "border-red-500/30 bg-red-500/10 text-red-400 hover:bg-red-500/20 hover:shadow-[0_0_15px_rgba(239,68,68,0.2)]"
            : "border-white/10 bg-white/5 text-neutral-300 hover:bg-white/10 hover:text-white hover:shadow-[0_0_15px_rgba(255,255,255,0.1)]"
        }`}
        aria-label={isVideoDisabled ? "Enable Camera" : "Disable Camera"}
      >
        {isVideoDisabled ? (
          <VideoOff className="h-5 w-5 transition-transform group-hover:scale-110" strokeWidth={1.5} />
        ) : (
          <Video className="h-5 w-5 transition-transform group-hover:scale-110" strokeWidth={1.5} />
        )}
      </button>
    </div>
  );
}