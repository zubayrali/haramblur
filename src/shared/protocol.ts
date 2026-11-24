import { defineExtensionMessaging } from "@webext-core/messaging";
import type { NudeNetLabel } from "./labels";

/** A single model detection; `box` is normalized 0..1 over the source media. */
export interface Detection {
	label: NudeNetLabel;
	score: number;
	box: [x: number, y: number, w: number, h: number];
}

export type VerdictDecision = "blur" | "clear";

export interface Verdict {
	decision: VerdictDecision;
	reasons: Detection[];
	ms: number;
	cached: boolean;
}

export type EngineState = "booting" | "ready" | "degraded" | "failed";
export type EngineBackend = "webgpu" | "wasm";

export interface EngineStatus {
	state: EngineState;
	backend: EngineBackend | null;
	error?: string;
}

export interface Pong {
	pong: true;
	ts: number;
}

/**
 * Typed protocol for every cross-context call in the extension
 * (docs/PLAN.md "Message protocol"). `ping` is a scaffold health check used by
 * the E2E smoke test.
 */
export interface HaramBlurProtocol {
	ping(): Pong;
	classifyImage(data: { url: string; width: number; height: number; frameUrl?: string }): Verdict;
	classifyFrame(data: { dataUrl: string; videoId: string; ts: number }): Verdict;
	engineStatus(): EngineStatus;
}

export const { sendMessage, onMessage } = defineExtensionMessaging<HaramBlurProtocol>();
