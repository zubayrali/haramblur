import type { Verdict } from "../../shared/protocol";
import type { Settings } from "../../shared/settings";

/** Arguments for the protocol's `classifyFrame` message. */
export interface ClassifyFrameArgs {
	dataUrl: string;
	videoId: string;
	ts: number;
}

export type ClassifyFrame = (args: ClassifyFrameArgs) => Promise<Verdict>;

export type FrameCallback = () => void;

/**
 * Minimal document shape the sampler needs. `Document` satisfies this in the
 * content script; tests pass a fake.
 */
export interface DocumentLike {
	readonly hidden: boolean;
	addEventListener(type: "visibilitychange", listener: () => void): void;
	removeEventListener(type: "visibilitychange", listener: () => void): void;
}

/** Minimal IntersectionObserver shape (real IO satisfies it in the browser). */
export interface IntersectionObserverLike {
	observe(target: Element): void;
	unobserve(target: Element): void;
	disconnect(): void;
}

export interface IntersectionEntryLike {
	readonly isIntersecting: boolean;
}

/** The canvas flavours `frames.ts` can draw onto. */
export type CanvasLike = OffscreenCanvas | HTMLCanvasElement;

export interface Canvas2dLike {
	drawImage(
		source: CanvasImageSource | HTMLVideoElement,
		dx: number,
		dy: number,
		dw: number,
		dh: number,
	): void;
	getImageData(sx: number, sy: number, sw: number, sh: number): { data: Uint8ClampedArray };
}

/**
 * Injectable seams so the sampler can run under a fake clock, a fake video
 * element and a fake document in unit tests, and inside a plain (non-extension)
 * page in the Chromium harness. Every field has a browser default.
 */
export interface SamplerEnvironment {
	/** Clock — `Date.now` in production, fake clock in tests. */
	now: () => number;
	setTimeout: (cb: () => void, ms: number) => unknown;
	clearTimeout: (handle: unknown) => void;
	/**
	 * Per-frame callback driver: `video.requestVideoFrameCallback` when the
	 * element has it, else `requestAnimationFrame`.
	 */
	requestFrame: (video: HTMLVideoElement, cb: FrameCallback) => number;
	cancelFrame: (video: HTMLVideoElement, handle: number) => void;
	/** `document` — its `hidden` flag and `visibilitychange` event. */
	document: DocumentLike;
	createIntersectionObserver: (
		cb: (entries: IntersectionEntryLike[]) => void,
	) => IntersectionObserverLike;
	/** Canvas factory: OffscreenCanvas when available, else a DOM canvas. */
	createCanvas: (width: number, height: number) => CanvasLike;
	/** Page hostname for the allowlist check (`location.hostname`). */
	hostname: () => string;
}

export interface VideoManagerDeps {
	classifyFrame: ClassifyFrame;
	getSettings: () => Promise<Settings>;
	watchSettings: (cb: (next: Settings, prev: Settings) => void) => () => void;
	env?: Partial<SamplerEnvironment>;
}

export interface VideoSamplerDeps {
	classifyFrame: ClassifyFrame;
	env?: Partial<SamplerEnvironment>;
	/**
	 * Called when the hysteresis-filtered blur decision for this video changes,
	 * or on a classify error (`verdict === null`, `error` set).
	 */
	onBlurChange?: (video: HTMLVideoElement, blurred: boolean, verdict: Verdict | null) => void;
	/**
	 * Called after every verdict, hysteresis applied or not. The manager uses
	 * it to reconcile a pre-verdict (fail-closed) blur once verdicts confirm
	 * clear — a state change alone never fires for an element that never
	 * flipped to blurred.
	 */
	onVerdict?: (video: HTMLVideoElement, verdict: Verdict) => void;
	onError?: (video: HTMLVideoElement, error: unknown) => void;
	/** Called whenever the machine's lifecycle state changes. */
	onStateChange?: (video: HTMLVideoElement, state: string) => void;
}

export function defaultEnvironment(): SamplerEnvironment {
	const doc = globalThis.document as unknown as DocumentLike;
	return {
		now: () => Date.now(),
		setTimeout: (cb, ms) => setTimeout(cb, ms),
		clearTimeout: (h) => clearTimeout(h as Parameters<typeof clearTimeout>[0]),
		requestFrame: (video, cb) => {
			if (typeof video.requestVideoFrameCallback === "function") {
				return video.requestVideoFrameCallback(cb);
			}
			return requestAnimationFrame(cb);
		},
		cancelFrame: (video, handle) => {
			if (typeof video.cancelVideoFrameCallback === "function") {
				video.cancelVideoFrameCallback(handle);
				return;
			}
			cancelAnimationFrame(handle);
		},
		document: doc,
		createIntersectionObserver: (cb) => new IntersectionObserver((entries) => cb(entries)),
		createCanvas: (width, height) => {
			if (typeof OffscreenCanvas !== "undefined") {
				return new OffscreenCanvas(width, height);
			}
			const canvas = globalThis.document.createElement("canvas");
			canvas.width = width;
			canvas.height = height;
			return canvas;
		},
		hostname: () => globalThis.location.hostname,
	};
}
