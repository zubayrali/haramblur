import { type AnyActorRef, assign, createActor, setup } from "xstate";
import type { Verdict, VerdictDecision } from "../../shared/protocol";
import { captureFrame, hashDistance } from "./frames";
import {
	defaultEnvironment,
	type IntersectionObserverLike,
	type SamplerEnvironment,
	type VideoSamplerDeps,
} from "./types";

// --- Adaptive rate (docs/PLAN.md: "default 2 fps, up to 6 fps after a scene
// change; pause while hidden, paused or off-screen") ---

export const BASE_FPS = 2;
export const BOOST_FPS = 6;
export const BOOST_MS = 5_000;
/** Once verdicts have agreed this many times in a row, drop to STABLE_FPS. */
export const STABLE_AFTER = 6;
export const STABLE_FPS = 1;
/** Normalized luma-hash distance above which a frame counts as a scene change. */
export const SCENE_CHANGE_DISTANCE = 0.3;

// --- Hysteresis ---

/** Consecutive agreeing verdicts required to flip blur on/off. */
export const HYSTERESIS_WINDOW = 2;
/** A single blur verdict with a detection this confident flips immediately. */
export const HIGH_CONFIDENCE_SCORE = 0.9;

export type SamplerState =
	| "idle"
	| "playing"
	| "sampling"
	| "paused"
	| "hidden"
	| "offscreen"
	| "ended";

export interface SamplerContext {
	videoId: string;
	/** Hysteresis-filtered blur state the DOM currently shows. */
	blurred: boolean;
	/** Recent raw verdict decisions, newest last (length ≤ HYSTERESIS_WINDOW). */
	decisions: VerdictDecision[];
	lastVerdict: Verdict | null;
	/** Consecutive identical verdict decisions (adaptive-rate input). */
	stableStreak: number;
}

export type SamplerEvent =
	| { type: "PLAYING" }
	| { type: "PAUSE" }
	| { type: "ENDED" }
	| { type: "DOC_HIDDEN" }
	| { type: "DOC_VISIBLE" }
	| { type: "OFF_SCREEN" }
	| { type: "ON_SCREEN" }
	| { type: "SLOT_GRANTED" }
	| { type: "SLOT_REVOKED" }
	| { type: "VERDICT"; verdict: Verdict }
	| { type: "DISPOSE" };

/** A verdict strong enough to flip blur on its own, without a second vote. */
export function isHighConfidence(verdict: Verdict): boolean {
	if (verdict.decision === "blur") {
		return verdict.reasons.some((r) => r.score >= HIGH_CONFIDENCE_SCORE);
	}
	return verdict.reasons.length === 0;
}

/**
 * Hysteresis reducer: blur flips only after HYSTERESIS_WINDOW consecutive
 * agreeing verdicts, or immediately on one high-confidence verdict
 * (docs/PLAN.md: "hysteresis so the blur doesn't flicker").
 */
export function applyVerdict(ctx: SamplerContext, verdict: Verdict): SamplerContext {
	const decisions = [...ctx.decisions, verdict.decision].slice(-HYSTERESIS_WINDOW);
	const stableStreak =
		ctx.decisions.length > 0 && ctx.decisions.at(-1) === verdict.decision
			? ctx.stableStreak + 1
			: 1;
	let blurred = ctx.blurred;
	if (
		!blurred &&
		verdict.decision === "blur" &&
		(isHighConfidence(verdict) ||
			(decisions.length === HYSTERESIS_WINDOW && decisions.every((d) => d === "blur")))
	) {
		blurred = true;
	} else if (
		blurred &&
		verdict.decision === "clear" &&
		(isHighConfidence(verdict) ||
			(decisions.length === HYSTERESIS_WINDOW && decisions.every((d) => d === "clear")))
	) {
		blurred = false;
	}
	return { ...ctx, blurred, decisions, lastVerdict: verdict, stableStreak };
}

/**
 * Lifecycle of one video element (docs/PLAN.md):
 * `idle → playing → sampling → (paused | hidden | offscreen | ended)`.
 *
 * The machine holds lifecycle + verdict bookkeeping; `VideoSampler` feeds it
 * DOM events and drives the frame loop while the snapshot is in `sampling`.
 */
export const samplerMachine = setup({
	types: {
		context: {} as SamplerContext,
		events: {} as SamplerEvent,
		input: {} as { videoId: string },
	},
}).createMachine({
	id: "hbVideoSampler",
	initial: "idle",
	context: ({ input }) => ({
		videoId: input.videoId,
		blurred: false,
		decisions: [],
		lastVerdict: null,
		stableStreak: 0,
	}),
	on: {
		VERDICT: {
			actions: assign(({ context, event }) => applyVerdict(context, event.verdict)),
		},
		DISPOSE: ".ended",
	},
	states: {
		idle: { on: { PLAYING: "playing" } },
		playing: {
			on: {
				SLOT_GRANTED: "sampling",
				PAUSE: "paused",
				DOC_HIDDEN: "hidden",
				OFF_SCREEN: "offscreen",
				ENDED: "ended",
			},
		},
		sampling: {
			on: {
				SLOT_REVOKED: "playing",
				PAUSE: "paused",
				DOC_HIDDEN: "hidden",
				OFF_SCREEN: "offscreen",
				ENDED: "ended",
			},
		},
		paused: {
			on: {
				PLAYING: "playing",
				DOC_HIDDEN: "hidden",
				OFF_SCREEN: "offscreen",
				ENDED: "ended",
			},
		},
		hidden: {
			on: {
				DOC_VISIBLE: "playing",
				OFF_SCREEN: "offscreen",
				PAUSE: "paused",
				ENDED: "ended",
			},
		},
		offscreen: {
			on: {
				ON_SCREEN: "playing",
				DOC_HIDDEN: "hidden",
				PAUSE: "paused",
				ENDED: "ended",
			},
		},
		ended: { type: "final" },
	},
});

export class VideoSampler {
	readonly videoId: string;
	readonly video: HTMLVideoElement;
	private readonly deps: VideoSamplerDeps;
	private readonly env: SamplerEnvironment;
	private readonly actor: AnyActorRef;
	private observer: IntersectionObserverLike | null = null;
	private readonly listeners: Array<() => void> = [];
	private intersecting = true;
	private frameHandle: number | null = null;
	private inFlight = false;
	private lastHash: Uint8Array | null = null;
	private lastFrameTs = -1;
	private lastSampleAt = 0;
	private boostUntil = 0;
	private rateFps = BASE_FPS;
	private lastBlurred = false;
	private lastState: SamplerState = "idle";
	private started = false;

	constructor(video: HTMLVideoElement, videoId: string, deps: VideoSamplerDeps) {
		this.video = video;
		this.videoId = videoId;
		this.deps = deps;
		this.env = { ...defaultEnvironment(), ...deps.env };
		this.actor = createActor(samplerMachine, { input: { videoId } });
	}

	get state(): string {
		return this.actor.getSnapshot().value;
	}

	/** Hysteresis-filtered blur state (what the DOM shows). */
	get blurred(): boolean {
		return this.actor.getSnapshot().context.blurred;
	}

	/** Recent raw verdict decisions, newest last (length ≤ HYSTERESIS_WINDOW). */
	get decisions(): readonly VerdictDecision[] {
		return this.actor.getSnapshot().context.decisions;
	}

	/** Most recent verdict, for settings re-decides that must not lose reasons. */
	get lastVerdict(): Verdict | null {
		return this.actor.getSnapshot().context.lastVerdict;
	}

	get sampling(): boolean {
		return this.state === "sampling";
	}

	/** Current adaptive sampling rate (fps) — exposed for tests/diagnostics. */
	get currentRateFps(): number {
		return this.rateFps;
	}

	start(): void {
		if (this.started) return;
		this.started = true;
		this.lastState = this.state as SamplerState;
		this.lastBlurred = this.blurred;
		this.actor.subscribe((snapshot) => this.onSnapshot(snapshot));
		this.actor.start();

		const on = (
			target: Pick<HTMLVideoElement, "addEventListener" | "removeEventListener">,
			type: string,
			fn: () => void,
		) => {
			target.addEventListener(type, fn);
			this.listeners.push(() => target.removeEventListener(type, fn));
		};
		on(this.video, "play", () => this.send({ type: "PLAYING" }));
		on(this.video, "playing", () => this.send({ type: "PLAYING" }));
		on(this.video, "pause", () => this.send({ type: "PAUSE" }));
		on(this.video, "ended", () => this.send({ type: "ENDED" }));
		this.env.document.addEventListener("visibilitychange", this.onVisibility);
		this.listeners.push(() =>
			this.env.document.removeEventListener("visibilitychange", this.onVisibility),
		);

		this.observer = this.env.createIntersectionObserver((entries) => {
			this.intersecting = entries.at(-1)?.isIntersecting ?? true;
			this.send({ type: this.intersecting ? "ON_SCREEN" : "OFF_SCREEN" });
		});
		this.observer.observe(this.video);

		// Reconcile attach-time truth (the video may already be playing).
		if (this.video.ended) this.send({ type: "ENDED" });
		else if (!this.video.paused) this.send({ type: "PLAYING" });
	}

	/** Manager grants one of the MAX_CONCURRENT_VIDEOS sampling slots. */
	grantSlot(): void {
		this.send({ type: "SLOT_GRANTED" });
	}

	revokeSlot(): void {
		this.send({ type: "SLOT_REVOKED" });
	}

	dispose(): void {
		this.send({ type: "DISPOSE" });
		this.stopLoop();
		this.observer?.disconnect();
		this.observer = null;
		for (const off of this.listeners.splice(0)) off();
		this.actor.stop();
	}

	private readonly onVisibility = () => {
		this.send({ type: this.env.document.hidden ? "DOC_HIDDEN" : "DOC_VISIBLE" });
	};

	private send(event: SamplerEvent): void {
		try {
			this.actor.send(event);
		} catch {
			// Actor already stopped.
		}
		// Reconcile pause: visibility/intersection transitions land on "playing"
		// even if the element has since paused.
		if (this.video.paused && this.state === "playing") {
			try {
				this.actor.send({ type: "PAUSE" });
			} catch {
				// stopped
			}
		}
	}

	private onSnapshot(snapshot: { value: string; context: SamplerContext }): void {
		const state = snapshot.value;
		if (state === "sampling") this.ensureLoop();
		else this.stopLoop();
		if (state !== this.lastState) {
			this.lastState = state as SamplerState;
			this.deps.onStateChange?.(this.video, state);
		}
		if (snapshot.context.blurred !== this.lastBlurred) {
			this.lastBlurred = snapshot.context.blurred;
			this.deps.onBlurChange?.(this.video, snapshot.context.blurred, snapshot.context.lastVerdict);
		}
	}

	private ensureLoop(): void {
		if (this.frameHandle === null) this.scheduleTick();
	}

	private stopLoop(): void {
		if (this.frameHandle !== null) {
			this.env.cancelFrame(this.video, this.frameHandle);
			this.frameHandle = null;
		}
	}

	private scheduleTick(): void {
		this.frameHandle = this.env.requestFrame(this.video, () => {
			this.frameHandle = null;
			this.tick();
		});
	}

	private tick(): void {
		if (this.state !== "sampling") return;
		// Belt-and-braces reconcile: events can be missed (e.g. synthetic videos).
		if (this.video.ended) {
			this.send({ type: "ENDED" });
			return;
		}
		if (this.video.paused) {
			this.send({ type: "PAUSE" });
			return;
		}
		if (this.env.document.hidden) {
			this.send({ type: "DOC_HIDDEN" });
			return;
		}
		if (!this.intersecting) {
			this.send({ type: "OFF_SCREEN" });
			return;
		}
		const now = this.env.now();
		if (now - this.lastSampleAt >= 1000 / this.rateFps) {
			void this.sampleOnce();
		}
		this.scheduleTick();
	}

	private async sampleOnce(): Promise<void> {
		if (this.inFlight) return; // one in-flight request per video
		this.inFlight = true;
		try {
			const frame = await captureFrame(this.video, this.env);
			const now = this.env.now();
			if (!frame) return;
			// First frame has nothing to compare against → not a scene change.
			const dist = this.lastHash ? hashDistance(frame.hash, this.lastHash) : 0;
			// Skip duplicate frames: frozen stream (currentTime unchanged) or a
			// pixel-identical frame — but never skip the very first capture.
			const duplicate = this.lastHash !== null && (frame.ts === this.lastFrameTs || dist === 0);
			this.lastHash = frame.hash;
			this.lastFrameTs = frame.ts;
			this.lastSampleAt = now;
			if (duplicate) {
				this.updateRate(dist);
				return;
			}
			const verdict = await this.deps.classifyFrame({
				dataUrl: frame.dataUrl,
				videoId: this.videoId,
				ts: frame.ts,
			});
			this.actor.send({ type: "VERDICT", verdict });
			this.deps.onVerdict?.(this.video, verdict);
			this.updateRate(dist);
		} catch (err) {
			this.deps.onError?.(this.video, err);
		} finally {
			this.inFlight = false;
		}
	}

	private updateRate(dist: number): void {
		const now = this.env.now();
		const streak = this.actor.getSnapshot().context.stableStreak;
		if (dist > SCENE_CHANGE_DISTANCE) {
			this.boostUntil = now + BOOST_MS;
			this.rateFps = BOOST_FPS;
		} else if (now < this.boostUntil) {
			this.rateFps = BOOST_FPS;
		} else if (streak >= STABLE_AFTER) {
			this.rateFps = STABLE_FPS;
		} else {
			this.rateFps = BASE_FPS;
		}
	}
}
