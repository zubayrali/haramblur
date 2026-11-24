import type { Verdict } from "../../shared/protocol";
import type { Settings } from "../../shared/settings";
import { VideoSampler } from "./sampler";
import { defaultEnvironment, type SamplerEnvironment, type VideoManagerDeps } from "./types";

/** Max videos sampling at once (docs/PLAN.md adaptive-rate budget). */
export const MAX_CONCURRENT_VIDEOS = 4;
/** Hover delay before a blurred video reveals, matching the image behaviour. */
export const REVEAL_DELAY_MS = 1_000;

/**
 * Coordinates one `VideoSampler` per `<video>`: admits videos, hands out at
 * most MAX_CONCURRENT_VIDEOS sampling slots, gates on settings (master switch,
 * `blurVideos`, per-site allowlist) and applies/removes the blur CSS.
 *
 * Discovery is WP2's job — `src/content/discovery.ts` calls `observe` /
 * `unobserve`; the queue's classify transport arrives through
 * `deps.classifyFrame` (usually `sendMessage("classifyFrame", ...)`).
 */
export class VideoManager {
	private readonly deps: VideoManagerDeps;
	private readonly env: SamplerEnvironment;
	private readonly samplers = new Map<HTMLVideoElement, VideoSampler>();
	private readonly revealCleanups = new Map<HTMLVideoElement, () => void>();
	private readonly revealTimers = new Map<HTMLVideoElement, unknown>();
	private counter = 0;
	private settings: Settings | null = null;
	private unwatch: (() => void) | null = null;
	private started = false;

	constructor(deps: VideoManagerDeps) {
		this.deps = deps;
		this.env = { ...defaultEnvironment(), ...deps.env };
	}

	get size(): number {
		return this.samplers.size;
	}

	samplerFor(video: HTMLVideoElement): VideoSampler | undefined {
		return this.samplers.get(video);
	}

	async start(): Promise<void> {
		if (this.started) return;
		this.started = true;
		this.settings = await this.deps.getSettings();
		this.unwatch = this.deps.watchSettings((next) => this.applySettings(next));
	}

	/** Whether video detection may run at all right now. */
	get allowed(): boolean {
		const s = this.settings;
		if (!s) return false;
		return s.status && s.blurVideos && !s.allowlist.includes(this.env.hostname());
	}

	observe(video: HTMLVideoElement): void {
		if (this.samplers.has(video)) return;
		if (!this.allowed) return;
		const sampler = new VideoSampler(video, `hb-video-${++this.counter}`, {
			classifyFrame: this.deps.classifyFrame,
			env: this.env,
			onBlurChange: (v, blurred, verdict) => this.onBlurChange(v, blurred, verdict),
			onError: (v, err) => this.onError(v, err),
			onStateChange: () => this.grantSlots(),
		});
		this.samplers.set(video, sampler);
		sampler.start();
		// Fail closed: blur until the first verdict when blurryStartMode is on.
		if (this.settings?.blurryStartMode) this.applyBlur(video, "processing");
		this.grantSlots();
	}

	unobserve(video: HTMLVideoElement): void {
		const sampler = this.samplers.get(video);
		if (!sampler) return;
		sampler.dispose();
		this.samplers.delete(video);
		this.clearBlur(video);
		this.grantSlots();
	}

	dispose(): void {
		this.unwatch?.();
		this.unwatch = null;
		for (const video of [...this.samplers.keys()]) this.unobserve(video);
	}

	/**
	 * Keep at most MAX_CONCURRENT_VIDEOS samplers in `sampling`; the rest wait
	 * in `playing` until a slot frees. Call after every state change.
	 */
	private grantSlots(): void {
		let active = 0;
		for (const s of this.samplers.values()) if (s.state === "sampling") active++;
		for (const s of this.samplers.values()) {
			if (active >= MAX_CONCURRENT_VIDEOS) break;
			if (s.state !== "playing") continue;
			s.grantSlot();
			if (s.sampling) active++;
		}
	}

	private onBlurChange(video: HTMLVideoElement, blurred: boolean, verdict: Verdict | null): void {
		if (blurred) this.applyBlur(video, "blurred", verdict);
		else this.clearBlur(video);
	}

	private onError(video: HTMLVideoElement, _err: unknown): void {
		// Fail mode (PLAN.md): keep blurred by default on timeout/error.
		if (this.settings?.failClosed ?? true) this.applyBlur(video, "error");
		else this.clearBlur(video);
	}

	private applyBlur(
		video: HTMLVideoElement,
		state: "processing" | "blurred" | "error",
		verdict?: Verdict | null,
	): void {
		const s = this.settings;
		const blurPx = Math.min(Math.max(s?.blurAmount ?? 20, 0), 100);
		video.dataset.hbState = state;
		video.dataset.hbBlur = "";
		video.dataset.hbReasons = JSON.stringify(verdict?.reasons ?? []);
		video.style.setProperty("--hb-blur", `${blurPx}px`);
		video.style.filter = `blur(${blurPx}px)${s?.gray ? " grayscale(1)" : ""}`;
		if (s?.unblurVideos) this.bindReveal(video);
		else this.unbindReveal(video);
	}

	private clearBlur(video: HTMLVideoElement): void {
		video.style.filter = "";
		delete video.dataset.hbState;
		delete video.dataset.hbBlur;
		delete video.dataset.hbReasons;
		delete video.dataset.hbRevealed;
		this.unbindReveal(video);
	}

	/**
	 * Reveal-on-hover, same settings as images (`unblurVideos`): after
	 * REVEAL_DELAY_MS hovering a blurred video, drop the filter until the
	 * pointer leaves.
	 */
	private bindReveal(video: HTMLVideoElement): void {
		if (this.revealCleanups.has(video)) return;
		const enter = () => {
			this.revealTimers.set(
				video,
				this.env.setTimeout(() => {
					video.style.filter = "";
					video.dataset.hbRevealed = "";
				}, REVEAL_DELAY_MS),
			);
		};
		const leave = () => {
			const timer = this.revealTimers.get(video);
			if (timer !== undefined) this.env.clearTimeout(timer);
			this.revealTimers.delete(video);
			if (video.dataset.hbRevealed === undefined) return;
			delete video.dataset.hbRevealed;
			const sampler = this.samplers.get(video);
			if (sampler?.blurred) this.applyBlur(video, "blurred", undefined);
		};
		video.addEventListener("mouseenter", enter);
		video.addEventListener("mouseleave", leave);
		this.revealCleanups.set(video, () => {
			video.removeEventListener("mouseenter", enter);
			video.removeEventListener("mouseleave", leave);
			const timer = this.revealTimers.get(video);
			if (timer !== undefined) this.env.clearTimeout(timer);
			this.revealTimers.delete(video);
		});
	}

	private unbindReveal(video: HTMLVideoElement): void {
		this.revealCleanups.get(video)?.();
		this.revealCleanups.delete(video);
	}

	private applySettings(next: Settings): void {
		this.settings = next;
		if (!this.allowed) {
			// Turned off (master/videos toggle or site allowlisted): stop every
			// sampler and unblur.
			for (const video of [...this.samplers.keys()]) this.unobserve(video);
			return;
		}
		// Settings like blurAmount/gray/unblurVideos apply live.
		for (const sampler of this.samplers.values()) {
			const video = sampler.video;
			if (sampler.blurred) this.applyBlur(video, "blurred", undefined);
			else this.unbindReveal(video);
		}
		this.grantSlots();
	}
}
