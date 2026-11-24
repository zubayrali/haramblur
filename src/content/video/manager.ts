import type { Verdict } from "../../shared/protocol";
import { DEFAULT_SETTINGS, type Settings } from "../../shared/settings";
import { isAllowlisted } from "../../shared/site";
import { blurOnError, blurWhilePending, setBlurred, setReasons, setState } from "../blur";
import { applyBlurMode, removeRegionBlur } from "../region";
import { reasonsToJson } from "../site";
import { HYSTERESIS_WINDOW, isHighConfidence, VideoSampler } from "./sampler";
import { defaultEnvironment, type SamplerEnvironment, type VideoManagerDeps } from "./types";

/** Max videos sampling at once (docs/PLAN.md adaptive-rate budget). */
export const MAX_CONCURRENT_VIDEOS = 4;

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
		// Same allowlist semantics as the content pipeline (site.ts): normalized
		// entries, subdomain match — a bare `includes` missed both.
		return s.status && s.blurVideos && !isAllowlisted(this.env.hostname(), s.allowlist);
	}

	observe(video: HTMLVideoElement): void {
		if (this.samplers.has(video)) return;
		// The pipeline can emit videos before start()'s getSettings resolves —
		// retry once settings are loaded instead of dropping the element.
		if (!this.started) {
			void this.start().then(() => this.observe(video));
			return;
		}
		if (!this.allowed) return;
		const sampler = new VideoSampler(video, `hb-video-${++this.counter}`, {
			classifyFrame: this.deps.classifyFrame,
			env: this.env,
			onBlurChange: (v, blurred, verdict) => this.onBlurChange(v, blurred, verdict),
			onVerdict: (v, verdict) => this.onVerdict(v, verdict),
			onError: (v, err) => this.onError(v, err),
			onStateChange: () => this.grantSlots(),
		});
		this.samplers.set(video, sampler);
		sampler.start();
		// Pending blur follows blurryStartMode — same rule as images; the
		// sampler's own verdicts take over from here.
		if (this.settings) this.applyBlurState(video, "pending", null);
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
		this.started = false;
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
		if (blurred) this.applyBlurState(video, "blur", verdict);
		else this.clearBlur(video);
	}

	/**
	 * Reconcile the DOM on every verdict, not just hysteresis flips: a
	 * pre-verdict (pending) blur must clear once clear is confirmed, and a
	 * blurred element clears when hysteresis flips back. While blurred, every
	 * verdict refreshes the region boxes — scene changes move them.
	 */
	private onVerdict(video: HTMLVideoElement, verdict: Verdict): void {
		const sampler = this.samplers.get(video);
		if (sampler == null) return;
		if (sampler.blurred) {
			this.applyBlurState(video, "blur", verdict);
			return;
		}
		const confirmedClear =
			isHighConfidence(verdict) ||
			(sampler.decisions.length >= HYSTERESIS_WINDOW &&
				sampler.decisions.every((d) => d === "clear"));
		if (confirmedClear) this.clearBlur(video);
	}

	private onError(video: HTMLVideoElement, _err: unknown): void {
		// Fail mode (PLAN.md): keep blurred by default on timeout/error.
		this.applyBlurState(video, "error", null);
	}

	/** Reveal a blurred video permanently (context menu / click reveal). */
	reveal(video: HTMLVideoElement): void {
		this.clearBlur(video);
		video.dataset.hbReveal = "click";
	}

	/**
	 * Apply the unified `data-hb-state` vocabulary ("pending" | "blur" |
	 * "clear" | "error") and route the visual through blur.ts + the region
	 * portal — no inline `style.filter`, so hover reveal stays a single
	 * mechanism shared with images.
	 */
	private applyBlurState(
		video: HTMLVideoElement,
		state: "pending" | "blur" | "error",
		verdict: Verdict | null,
	): void {
		// A user-initiated reveal wins over later verdicts/settings changes.
		if (video.dataset.hbReveal === "click") return;
		const s = this.settings;
		setState(video, state);
		setReasons(video, reasonsToJson(verdict?.reasons ?? []));
		if (state === "blur" && verdict != null && s != null) {
			applyBlurMode(video, verdict.reasons, "video", s);
			return;
		}
		removeRegionBlur(video);
		const effective = s ?? DEFAULT_SETTINGS;
		setBlurred(video, state === "pending" ? blurWhilePending(effective) : blurOnError(effective));
	}

	private clearBlur(video: HTMLVideoElement): void {
		removeRegionBlur(video);
		setBlurred(video, false);
		// Keep a hb-state so getTabStats still counts the video (as "clear").
		setState(video, "clear");
		delete video.dataset.hbReasons;
		delete video.dataset.hbRevealed;
	}

	private applySettings(next: Settings): void {
		this.settings = next;
		if (!this.allowed) {
			// Turned off (master/videos toggle or site allowlisted): stop every
			// sampler and unblur.
			for (const video of [...this.samplers.keys()]) this.unobserve(video);
			return;
		}
		// Settings like blurAmount/gray/unblurVideos/regionBlur apply live —
		// and the blur must be rebuilt from the sampler's last verdict, not
		// wiped: reasons drive both the policy re-decide and region boxes.
		for (const sampler of this.samplers.values()) {
			const video = sampler.video;
			if (sampler.blurred) {
				this.applyBlurState(video, "blur", sampler.lastVerdict);
			} else if (video.dataset.hbState === "pending" || video.dataset.hbState === "error") {
				this.applyBlurState(video, video.dataset.hbState as "pending" | "error", null);
			} else {
				removeRegionBlur(video);
			}
		}
		this.grantSlots();
	}
}
