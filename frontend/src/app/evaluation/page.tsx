"use client";

import { useEffect, useState, useCallback } from "react";
import type { ReactElement } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, RefreshCw, ShieldCheck, TriangleAlert } from "lucide-react";

import { LLMMarkdown } from "@/components/feedback/LLMMarkdown";
import type { CoachingFeedbackPayload } from "@/types/backend";

/**
 * Renders the post-interview diagnostic evaluation screen.
 * Retrieves the cached LLM payload from the local session and orchestrates the Markdown display.
 *
 * @returns {ReactElement} The high-tech evaluation dashboard.
 */
export default function EvaluationResult(): ReactElement {
  const router = useRouter();
  const [evaluation, setEvaluation] = useState<CoachingFeedbackPayload | null>(null);
  const [hasLoaded, setHasLoaded] = useState<boolean>(false);
  const [parseError, setParseError] = useState<string | null>(null);

  useEffect(() => {
    try {
      const cachedData = sessionStorage.getItem("coaching_evaluation");
      if (cachedData) {
        const parsedPayload = JSON.parse(cachedData) as CoachingFeedbackPayload;
        setEvaluation(parsedPayload);
      }
    } catch (error) {
      setParseError(error instanceof Error ? error.message : "Failed to decode evaluation payload.");
    } finally {
      setHasLoaded(true);
    }
  }, []);

  const handleReturnHome = useCallback((): void => {
    sessionStorage.removeItem("coaching_evaluation");
    router.push("/");
  }, [router]);

  const handleRetryInterview = useCallback((): void => {
    sessionStorage.removeItem("coaching_evaluation");
    router.push("/interview");
  }, [router]);

  if (!hasLoaded) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-neutral-950">
        <div className="flex flex-col items-center gap-4">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-cyan-500 border-t-transparent" />
          <span className="font-mono text-xs uppercase tracking-widest text-cyan-500">
            Decrypting Matrix...
          </span>
        </div>
      </main>
    );
  }

  if (parseError || !evaluation) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-neutral-950 p-6">
        <div className="flex max-w-md flex-col items-center gap-6 rounded-2xl border border-red-500/20 bg-red-950/10 p-8 text-center backdrop-blur-md">
          <TriangleAlert className="h-12 w-12 text-red-500" strokeWidth={1.5} />
          <div className="flex flex-col gap-2">
            <h2 className="font-mono text-lg font-bold uppercase tracking-widest text-red-400">
              Telemetry Lost
            </h2>
            <p className="font-mono text-sm text-red-300/80">
              {parseError || "No evaluation data found in the current session matrix."}
            </p>
          </div>
          <button
            onClick={handleReturnHome}
            type="button"
            className="mt-4 flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-6 py-3 font-mono text-xs font-semibold uppercase tracking-widest text-red-400 transition-colors hover:bg-red-500/20"
          >
            <ArrowLeft className="h-4 w-4" />
            Return to Base
          </button>
        </div>
      </main>
    );
  }

  return (
    <main className="relative flex min-h-screen flex-col bg-neutral-950 text-neutral-200 selection:bg-emerald-500/30">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-emerald-900/10 via-neutral-950 to-neutral-950" />
      <div className="pointer-events-none fixed inset-0 bg-[url('/assets/grid.svg')] bg-center opacity-5" />

      <header className="relative z-10 flex h-20 w-full items-center justify-between border-b border-white/5 bg-black/40 px-6 backdrop-blur-xl sm:px-10">
        <div className="flex items-center gap-3">
          <ShieldCheck className="h-6 w-6 text-emerald-400" strokeWidth={2} />
          <h1 className="font-mono text-base font-bold uppercase tracking-widest text-white">
            Performance Report
          </h1>
        </div>
        <div className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-4 py-1.5 font-mono text-[10px] uppercase tracking-widest text-emerald-400">
          Segment ID: {evaluation.segment_id.toString().padStart(4, "0")}
        </div>
      </header>

      <section className="relative z-10 flex flex-1 flex-col items-center px-4 py-10 sm:px-6 lg:px-8">
        <div className="w-full max-w-4xl">
          <LLMMarkdown content={evaluation.feedback_text} className="w-full shadow-2xl" />

          <div className="mt-8 flex flex-col items-center justify-end gap-4 sm:flex-row">
            <button
              onClick={handleReturnHome}
              type="button"
              className="flex w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 px-6 py-4 font-mono text-xs font-semibold uppercase tracking-widest text-neutral-300 transition-all hover:bg-white/10 hover:text-white sm:w-auto"
            >
              <ArrowLeft className="h-4 w-4" />
              Main Menu
            </button>
            <button
              onClick={handleRetryInterview}
              type="button"
              className="group relative flex w-full items-center justify-center gap-2 overflow-hidden rounded-xl border border-emerald-500/50 bg-emerald-500/10 px-8 py-4 font-mono text-xs font-bold uppercase tracking-widest text-emerald-400 transition-all hover:border-emerald-400 hover:bg-emerald-500/20 hover:shadow-[0_0_30px_rgba(16,185,129,0.2)] sm:w-auto"
            >
              <div className="absolute inset-0 translate-x-[-100%] bg-gradient-to-r from-emerald-500/0 via-emerald-500/10 to-emerald-500/0 transition-transform duration-700 group-hover:translate-x-[100%]" />
              <RefreshCw className="z-10 h-4 w-4 transition-transform group-hover:rotate-180" />
              <span className="z-10">Deploy New Session</span>
            </button>
          </div>
        </div>
      </section>
    </main>
  );
}