"use client";

import { useCallback, useState } from "react";
import type { ReactElement } from "react";
import { useRouter } from "next/navigation";
import { AlertCircle, Bot, Cpu, Play, Terminal } from "lucide-react";

/**
 * Serves as the primary entry point for the application, establishing the simulation context
 * and brokering navigation to the active interview environment.
 *
 * @returns {ReactElement} The high-tech landing page layout and interaction surface.
 */
export default function LandingPage(): ReactElement {
  const router = useRouter();
  const [isRouting, setIsRouting] = useState<boolean>(false);
  const [routeError, setRouteError] = useState<string | null>(null);

  const handleStartInterview = useCallback(async (): Promise<void> => {
    if (isRouting) {
      return;
    }

    try {
      setIsRouting(true);
      setRouteError(null);
      router.push("/interview");
    } catch (error) {
      setRouteError(
        error instanceof Error ? error.message : "Failed to initialize the interview environment."
      );
      setIsRouting(false);
    }
  }, [isRouting, router]);

  return (
    <main className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-neutral-950 p-6 text-neutral-200">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-cyan-900/20 via-neutral-950 to-neutral-950" />
      <div className="pointer-events-none absolute inset-0 bg-[url('/assets/grid.svg')] bg-center opacity-10" />

      <div className="relative z-10 flex w-full max-w-2xl flex-col items-center gap-8 rounded-3xl border border-white/5 bg-black/40 p-10 shadow-2xl backdrop-blur-2xl ring-1 ring-white/10">
        <div className="flex flex-col items-center gap-3 text-center">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-cyan-500/30 bg-cyan-500/10 text-cyan-400 shadow-[0_0_30px_rgba(34,211,238,0.2)]">
            <Cpu className="h-8 w-8" strokeWidth={1.5} />
          </div>
          <h1 className="mt-4 font-mono text-3xl font-bold tracking-tighter text-white sm:text-4xl">
            Multimodal Coaching Matrix
          </h1>
          <p className="max-w-md font-mono text-xs uppercase tracking-widest text-neutral-400">
            Behavioral analysis & semantic extraction
          </p>
        </div>

        <div className="w-full rounded-xl border border-white/10 bg-neutral-900/50 p-6 shadow-inner">
          <div className="mb-4 flex items-center gap-2 border-b border-white/10 pb-3">
            <Terminal className="h-4 w-4 text-emerald-400" />
            <span className="font-mono text-xs font-semibold uppercase tracking-widest text-emerald-400">
              Active Simulation Parameter
            </span>
          </div>
          <div className="flex items-start gap-4">
            <Bot className="mt-1 h-6 w-6 shrink-0 text-cyan-500" strokeWidth={1.5} />
            <p className="font-mono text-lg leading-relaxed text-neutral-200">
              &quot;Welcome to your evaluation. To begin, please tell us a little bit about yourself.&quot;
            </p>
          </div>
        </div>

        {routeError && (
          <div className="flex w-full items-center gap-3 rounded-lg border border-red-500/30 bg-red-950/30 p-4 text-red-400">
            <AlertCircle className="h-5 w-5 shrink-0" />
            <span className="font-mono text-xs">{routeError}</span>
          </div>
        )}

        <button
          onClick={handleStartInterview}
          disabled={isRouting}
          type="button"
          className="group relative flex h-14 w-full items-center justify-center gap-3 rounded-xl border border-cyan-500/50 bg-cyan-500/10 overflow-hidden transition-all duration-500 hover:border-cyan-400 hover:bg-cyan-500/20 hover:shadow-[0_0_40px_rgba(34,211,238,0.3)] disabled:cursor-not-allowed disabled:opacity-50 sm:w-2/3"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/0 via-cyan-500/10 to-cyan-500/0 translate-x-[-100%] transition-transform duration-1000 group-hover:translate-x-[100%]" />
          <Play className="z-10 h-5 w-5 fill-cyan-400 text-cyan-400 transition-transform group-hover:scale-110" />
          <span className="z-10 font-mono text-sm font-bold uppercase tracking-widest text-cyan-50">
            {isRouting ? "Initializing Node..." : "Start Interview Sequence"}
          </span>
        </button>
      </div>
    </main>
  );
}