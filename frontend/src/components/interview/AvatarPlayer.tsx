"use client";

import { useCallback, useRef, useState } from "react";
import type { ReactElement, SyntheticEvent } from "react";
import { AlertCircle } from "lucide-react";

export interface AvatarPlayerProps {
  /** Callback triggered immediately when the asking sequence completes. */
  onQuestionComplete: () => void;
  /** Optional Tailwind class string for container styling injection. */
  className?: string;
}

/**
 * Orchestrates the seamless playback transition between the active questioning video
 * and the continuous listening loop for the AI avatar.
 *
 * @param {AvatarPlayerProps} props - Component properties.
 * @returns {ReactElement} The rendered video player component.
 */
export function AvatarPlayer({
  onQuestionComplete,
  className = "",
}: AvatarPlayerProps): ReactElement {
  const [activePhase, setActivePhase] = useState<"asking" | "listening">("asking");
  const [playbackError, setPlaybackError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const handleSequenceEnd = useCallback(() => {
    if (activePhase === "asking") {
      setActivePhase("listening");
      onQuestionComplete();
    }
  }, [activePhase, onQuestionComplete]);

  const handleMediaError = useCallback((event: SyntheticEvent<HTMLVideoElement, Event>) => {
    const errorDetails = (event.target as HTMLVideoElement).error?.message || "Unknown media error";
    setPlaybackError(`Avatar stream interrupted: ${errorDetails}`);
  }, []);

  if (playbackError) {
    return (
      <div
        className={`flex h-full w-full flex-col items-center justify-center rounded-2xl border border-red-500/30 bg-red-950/20 p-6 text-center shadow-[0_0_30px_rgba(239,68,68,0.1)] backdrop-blur-md ${className}`}
        role="alert"
      >
        <AlertCircle className="mb-4 h-10 w-10 text-red-500" strokeWidth={1.5} />
        <span className="font-mono text-sm tracking-tight text-red-400">{playbackError}</span>
      </div>
    );
  }

  return (
    <div
      className={`relative overflow-hidden rounded-2xl border border-white/10 bg-black/80 shadow-[0_0_40px_rgba(0,0,0,0.5)] ring-1 ring-white/5 transition-all duration-700 ease-in-out ${className}`}
    >
      <video
        ref={videoRef}
        src={`/videos/avatar-${activePhase}.mp4`}
        autoPlay
        muted={activePhase === "listening"} 
        playsInline
        loop={activePhase === "listening"}
        onEnded={handleSequenceEnd}
        onError={handleMediaError}
        className="h-full w-full object-cover transition-opacity duration-500"
      />

      <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-black/10" />
      
      <div className="absolute bottom-4 left-4 flex items-center gap-3">
        <div className="relative flex h-3 w-3 items-center justify-center">
          {activePhase === "asking" && (
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-cyan-400 opacity-75" />
          )}
          <span
            className={`relative inline-flex h-2 w-2 rounded-full ${
              activePhase === "asking" ? "bg-cyan-500" : "bg-emerald-500"
            }`}
          />
        </div>
        <span className="font-mono text-xs font-medium uppercase tracking-widest text-white/80">
          {activePhase === "asking" ? "Avatar Active" : "Avatar Listening"}
        </span>
      </div>
    </div>
  );
}