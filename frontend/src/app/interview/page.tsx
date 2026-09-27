"use client";

import { useEffect, useCallback, useMemo, useState } from "react";
import type { ReactElement } from "react";
import { useRouter } from "next/navigation";
import { Activity, Network, ServerCrash } from "lucide-react";

import { useWebcamStream } from "@/hooks/useWebcamStream";
import { useGcpSocket } from "@/hooks/useGcpSocket";
import { AvatarPlayer } from "@/components/interview/AvatarPlayer";
import { UserWebcam } from "@/components/interview/UserWebcam";
import { RecordingControls } from "@/components/interview/RecordingControls";

/**
 * Encapsulates the live interview environment, managing the orchestration between
 * local hardware capture, the AI avatar video sequence, and the GCP WebSocket telemetry.
 *
 * @returns {ReactElement} The high-fidelity, real-time interview room interface.
 */
export default function InterviewRoom(): ReactElement {
  const router = useRouter();

  const {
    stream,
    isAudioMuted,
    isVideoDisabled,
    error: hardwareError,
    toggleAudio,
    toggleVideo,
    stopStream,
  } = useWebcamStream();

  const {
    connectionState,
    modalityStatus,
    evaluation,
    error: networkError,
    connect,
    startRecording,
    stopRecording,
    disconnect,
  } = useGcpSocket(stream);

  useEffect(() => {
    connect();
    return () => {
      disconnect();
      stopStream();
    };
  }, [connect, disconnect, stopStream]);

  useEffect(() => {
    if (connectionState === "EVALUATION_READY" && evaluation) {
      try {
        sessionStorage.setItem("coaching_evaluation", JSON.stringify(evaluation));
        router.push("/evaluation");
      } catch (e) {
        console.error("Failed to persist evaluation payload to session storage.", e);
      }
    }
  }, [connectionState, evaluation, router]);

  const handleQuestionComplete = useCallback((): void => {
    if (connectionState === "CONNECTED" && stream) {
      startRecording();
    }
  }, [connectionState, stream, startRecording]);

  const combinedError = useMemo(
    () => hardwareError?.message || networkError,
    [hardwareError, networkError]
  );

  const [hasFinishedRecording, setHasFinishedRecording] = useState(false);

  const handleEndAnswer = useCallback((): void => {
    setHasFinishedRecording(true);
    stopRecording();
  }, [stopRecording]);

  const isProcessing = hasFinishedRecording || connectionState === "PROCESSING" || connectionState === "EVALUATION_READY";

  return (
    <main className="relative flex min-h-screen flex-col bg-neutral-950 text-neutral-200 selection:bg-cyan-500/30">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_bottom,_var(--tw-gradient-stops))] from-cyan-900/10 via-neutral-950 to-neutral-950" />
      <div className="pointer-events-none absolute inset-0 bg-[url('/assets/grid.svg')] bg-center opacity-5" />

      <header className="relative z-10 flex h-16 w-full items-center justify-between border-b border-white/5 bg-black/40 px-6 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <Activity className="h-5 w-5 text-cyan-400" strokeWidth={2} />
          <h1 className="font-mono text-sm font-bold uppercase tracking-widest text-white">
            Active Session
          </h1>
        </div>

        <div className="flex items-center gap-6">
          {modalityStatus && (
            <div className="hidden items-center gap-3 md:flex">
              <TelemetryBadge label="Semantic" isActive={modalityStatus.semantic} />
              <TelemetryBadge label="Acoustic" isActive={modalityStatus.acoustic} />
              <TelemetryBadge label="Facial" isActive={modalityStatus.facial} />
              <TelemetryBadge label="Skeletal" isActive={modalityStatus.skeletal} />
            </div>
          )}

          <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 font-mono text-[10px] uppercase tracking-widest">
            <Network className="h-3 w-3 text-neutral-400" />
            <span
              className={`${
                connectionState === "ERROR"
                  ? "text-red-400"
                  : connectionState === "RECORDING"
                  ? "text-emerald-400"
                  : isProcessing
                  ? "text-cyan-400 animate-pulse"
                  : "text-neutral-300"
              }`}
            >
              {connectionState.replace("_", " ")}
            </span>
          </div>
        </div>
      </header>

      <section className="relative z-10 flex flex-1 flex-col gap-6 p-6 lg:flex-row">
        {combinedError ? (
          <div className="flex flex-1 flex-col items-center justify-center rounded-2xl border border-red-500/20 bg-red-950/10 p-8 shadow-2xl backdrop-blur-sm">
            <ServerCrash className="mb-4 h-12 w-12 text-red-500" strokeWidth={1.5} />
            <h2 className="font-mono text-lg font-bold uppercase tracking-widest text-red-400">
              System Fault
            </h2>
            <p className="mt-2 max-w-md text-center font-mono text-sm text-red-300/80">
              {combinedError}
            </p>
          </div>
        ) : isProcessing ? (
          <div className="flex flex-1 flex-col items-center justify-center rounded-2xl border border-cyan-500/20 bg-cyan-950/10 p-8 shadow-2xl backdrop-blur-sm">
            <div className="relative mb-6">
              <div className="absolute inset-0 animate-ping rounded-full border-2 border-cyan-400 opacity-20"></div>
              <Activity className="h-12 w-12 animate-pulse text-cyan-400" strokeWidth={1.5} />
            </div>
            <h2 className="font-mono text-lg font-bold uppercase tracking-widest text-cyan-400">
              Compiling AI Evaluation
            </h2>
            <p className="mt-2 max-w-md text-center font-mono text-sm text-cyan-300/80">
              Please wait while we analyze your performance and generate a detailed coaching report. This may take a few moments.
            </p>
          </div>
        ) : (
          <>
            <div className="flex flex-1 flex-col rounded-2xl">
              <AvatarPlayer
                onQuestionComplete={handleQuestionComplete}
                className="h-[40vh] w-full lg:h-full"
              />
            </div>

            <div className="flex flex-col gap-6 lg:w-[400px]">
              <UserWebcam
                stream={stream}
                isAudioMuted={isAudioMuted}
                isVideoDisabled={isVideoDisabled}
                error={hardwareError}
                className="aspect-video w-full shrink-0"
              />

              <div className="flex flex-1 flex-col justify-end">
                <RecordingControls
                  isRecording={connectionState === "RECORDING"}
                  isProcessing={isProcessing}
                  isAudioMuted={isAudioMuted}
                  isVideoDisabled={isVideoDisabled}
                  onToggleAudio={toggleAudio}
                  onToggleVideo={toggleVideo}
                  onEndAnswer={handleEndAnswer}
                />
              </div>
            </div>
          </>
        )}
      </section>
    </main>
  );
}

interface TelemetryBadgeProps {
  label: string;
  isActive: boolean;
}

function TelemetryBadge({ label, isActive }: TelemetryBadgeProps): ReactElement {
  return (
    <div className="flex items-center gap-1.5">
      <span className="relative flex h-2 w-2">
        {isActive && (
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-cyan-400 opacity-75" />
        )}
        <span
          className={`relative inline-flex h-2 w-2 rounded-full ${
            isActive ? "bg-cyan-500" : "bg-neutral-700"
          }`}
        />
      </span>
      <span
        className={`font-mono text-[9px] uppercase tracking-widest ${
          isActive ? "text-cyan-400" : "text-neutral-500"
        }`}
      >
        {label}
      </span>
    </div>
  );
}