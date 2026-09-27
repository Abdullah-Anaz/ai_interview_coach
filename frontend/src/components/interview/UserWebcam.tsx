"use client";

import { useEffect, useRef } from "react";
import type { ReactElement } from "react";
import { CameraOff, Mic, MicOff, Video, VideoOff } from "lucide-react";

export interface UserWebcamProps {
  /** Active media stream originating from the local user's hardware. */
  stream: MediaStream | null;
  /** Denotes whether audio tracks are currently transmitting data. */
  isAudioMuted: boolean;
  /** Denotes whether video tracks are currently transmitting frames. */
  isVideoDisabled: boolean;
  /** Explicit error state propagated from device permission or binding rejections. */
  error: Error | null;
  /** Optional Tailwind class string for container dimensions and positioning. */
  className?: string;
}

/**
 * Mounts and renders the client-side webcam stream with diagnostic telemetry overlays.
 *
 * @param {UserWebcamProps} props - Component configuration and media stream handles.
 * @returns {ReactElement} The visual webcam preview component.
 */
export function UserWebcam({
  stream,
  isAudioMuted,
  isVideoDisabled,
  error,
  className = "",
}: UserWebcamProps): ReactElement {
  const videoElementRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    const videoNode = videoElementRef.current;
    if (!videoNode) {
      return;
    }

    if (stream) {
      videoNode.srcObject = stream;
    } else {
      videoNode.srcObject = null;
    }

    return () => {
      if (videoNode) {
        videoNode.srcObject = null;
      }
    };
  }, [stream]);

  if (error) {
    return (
      <div
        className={`relative flex flex-col items-center justify-center overflow-hidden rounded-xl border border-red-500/30 bg-neutral-950/90 p-4 text-center shadow-lg backdrop-blur-md ${className}`}
        role="alert"
      >
        <CameraOff className="mb-2 h-8 w-8 text-red-400" strokeWidth={1.5} />
        <p className="font-mono text-xs uppercase tracking-wider text-red-400">
          Hardware Failure
        </p>
        <span className="mt-1 font-mono text-[10px] text-neutral-400">
          {error.message}
        </span>
      </div>
    );
  }

  return (
    <div
      className={`group relative overflow-hidden rounded-xl border border-white/10 bg-neutral-950/80 shadow-2xl backdrop-blur-xl ring-1 ring-white/5 transition-all duration-300 hover:border-cyan-500/30 hover:ring-cyan-500/20 ${className}`}
    >
      <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-cyan-500/10 via-transparent to-transparent" />

      {isVideoDisabled || !stream ? (
        <div className="flex h-full w-full flex-col items-center justify-center bg-neutral-900/60 text-neutral-500">
          <VideoOff className="h-8 w-8 stroke-[1.2]" />
          <span className="mt-2 font-mono text-[10px] tracking-widest uppercase text-neutral-400">
            Feed Suspended
          </span>
        </div>
      ) : (
        <video
          ref={videoElementRef}
          autoPlay
          playsInline
          muted
          className="h-full w-full -scale-x-100 object-cover"
        />
      )}

      <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-center justify-between bg-gradient-to-t from-black/80 via-black/40 to-transparent p-3">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span
              className={`absolute inline-flex h-full w-full rounded-full opacity-75 ${
                stream ? "animate-ping bg-emerald-400" : "bg-neutral-600"
              }`}
            />
            <span
              className={`relative inline-flex h-2 w-2 rounded-full ${
                stream ? "bg-emerald-500" : "bg-neutral-600"
              }`}
            />
          </span>
          <span className="font-mono text-[10px] uppercase tracking-widest text-neutral-300">
            {stream ? "Candidate Ingress" : "Standby"}
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <div
            className={`rounded-md p-1 backdrop-blur-md transition-colors ${
              isAudioMuted
                ? "bg-red-500/20 text-red-400 ring-1 ring-red-500/30"
                : "bg-black/40 text-neutral-300 ring-1 ring-white/10"
            }`}
          >
            {isAudioMuted ? (
              <MicOff className="h-3.5 w-3.5" />
            ) : (
              <Mic className="h-3.5 w-3.5" />
            )}
          </div>
          <div
            className={`rounded-md p-1 backdrop-blur-md transition-colors ${
              isVideoDisabled
                ? "bg-red-500/20 text-red-400 ring-1 ring-red-500/30"
                : "bg-black/40 text-neutral-300 ring-1 ring-white/10"
            }`}
          >
            {isVideoDisabled ? (
              <VideoOff className="h-3.5 w-3.5" />
            ) : (
              <Video className="h-3.5 w-3.5" />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}