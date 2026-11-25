import { defineExtensionMessaging } from "@webext-core/messaging";
import type { DetectionLabel } from "./labels";
import type { ModelSlot } from "./model-slots";
import type { Settings } from "./settings";
import type { WebGpuProbeReport } from "./webgpu-probe";

/** Binary person-instance mask, normalized over the source media. */
export interface PersonMask {
	/** Source-normalized rect the mask bitmap covers (the clipped instance box). */
	box: [x: number, y: number, w: number, h: number];
	width: number;
	height: number;
	/** Base64 MSB-first bit-packed alpha, row-major (`shared/mask.ts`). */
	bits: string;
	/** Inside-silhouette fraction of the mask bitmap, 0..1 when known. */
	coverage?: number;
}

/** A single model detection; `box` is normalized 0..1 over the source media. */
export interface Detection {
	label: DetectionLabel;
	score: number;
	box: [x: number, y: number, w: number, h: number];
	/** Estimated age in years when a face classifier provided one. */
	age?: number;
	/** Person silhouette associated with a FACE_FEMALE detection. */
	mask?: PersonMask;
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
	/** Per-model backend selections when the worker reported them. */
	backends?: Partial<Record<ModelSlot, EngineBackend>>;
	error?: string;
}

/**
 * Returned instead of a `Verdict` when the engine cannot answer — the caller
 * applies the user's fail mode (default fail closed) (docs/PLAN.md).
 */
export interface EngineError {
	error: string;
}

export type ClassifyResult = Verdict | EngineError;

export function isEngineError(result: ClassifyResult): result is EngineError {
	return typeof (result as EngineError).error === "string";
}

/**
 * A classify request forwarded from the background to the engine host.
 * `settings` travels with the request: offscreen documents cannot touch
 * `chrome.storage`, so the background reads settings for every call.
 */
export type EngineRequest =
	| {
			kind: "image";
			url: string;
			width: number;
			height: number;
			frameUrl?: string;
			settings?: Settings;
	  }
	| { kind: "frame"; dataUrl: string; videoId: string; ts: number; settings?: Settings };

export interface Pong {
	pong: true;
	ts: number;
}

/** Per-tab media bookkeeping reported to the popup, counted from `data-hb-state`. */
export interface TabStats {
	total: number;
	images: number;
	videos: number;
	/** Counts keyed by the element's `data-hb-state` value. */
	byState: Record<string, number>;
}

export type ContextMediaKind = "image" | "video";

/** Media targeted by a context-menu action, identified by its source URL. */
export interface MediaTarget {
	srcUrl: string;
	kind: ContextMediaKind;
}

/**
 * Typed protocol for every cross-context call in the extension
 * (docs/PLAN.md "Message protocol"). `ping` is a scaffold health check used by
 * the E2E smoke test.
 *
 * `revealMedia`/`recheckMedia` are sent to a content script via
 * `sendMessage(name, data, { tabId })`; the content pipeline (WP2) registers
 * the handlers.
 *
 * `classifyImage`/`classifyFrame` return a `Verdict`; when the engine cannot
 * answer the background throws instead, which `sendMessage` surfaces as a
 * rejection — WP2's content client applies `settings.failClosed` to it.
 * `engineRun`/`engineHostStatus` are internal: only the engine host registers
 * them; the background forwards public requests to it.
 */
export interface HaramBlurProtocol {
	ping(): Pong;
	classifyImage(data: { url: string; width: number; height: number; frameUrl?: string }): Verdict;
	classifyFrame(data: { dataUrl: string; videoId: string; ts: number }): Verdict;
	engineStatus(): EngineStatus;
	/** DP0 measurement probe: public entrypoint answered by the background. */
	webgpuProbe(): WebGpuProbeReport;
	getTabStats(data: { tabId: number }): TabStats;
	revealMedia(data: MediaTarget): void;
	recheckMedia(data: MediaTarget): void;
	engineRun(request: EngineRequest): ClassifyResult;
	engineHostStatus(): EngineStatus;
	/** Internal background → offscreen/host forwarding for `webgpuProbe`. */
	engineHostWebgpuProbe(): WebGpuProbeReport;
}

export const { sendMessage, onMessage } = defineExtensionMessaging<HaramBlurProtocol>();
