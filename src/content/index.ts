import { decide } from "../shared/policy";
import { onMessage, type Verdict } from "../shared/protocol";
import { getSettings, type Settings, watchSettings } from "../shared/settings-store";
import { isAllowlisted } from "../shared/site";
import {
	applyBlurStyles,
	blurOnError,
	blurWhilePending,
	clearAll,
	ensureBlurStyles,
	HOLD_PEEK_DELAY_MS,
	reveal,
	revealModeFor,
	setBlurred,
	setKind,
	setPeeking,
	setReasons,
	setState,
} from "./blur";
import {
	candidateFor,
	type DiscoveryHandle,
	deepQueryAll,
	type MediaCandidate,
	type MediaKind,
	startDiscovery,
} from "./discovery";
import { classifyImage } from "./engine";
import { captureMediaFrame } from "./frames";
import { TaskQueue } from "./queue";
import {
	applyBlurMode,
	clearAllRegionBlurs,
	peekRegionBlur,
	removeRegionBlur,
	setRegionSettings,
	syncRegionBlur,
} from "./region";
import { reasonsFromJson, reasonsToJson } from "./site";
import { createVideoManager, type VideoManager } from "./video";
import { applyYouTubeShields } from "./youtube";

/**
 * Content pipeline glue (docs/PLAN.md WP2). Deliberately small: discovery emits
 * candidates, the queue classifies them through the engine client, and blur.ts
 * owns the styling. No engine imports — only the `src/shared/` protocol.
 */

const QUEUE_CONCURRENCY = 4;
/** Elements at or below this size on both axes skip the pending pulse —
 * they're UI icons/emojis, not content worth fail-closed flashing. */
const TINY_ICON_MAX_PX = 32;

export async function startContentPipeline(): Promise<void> {
	let settings = await getSettings();
	ensureBlurStyles();
	applyBlurStyles(settings);
	setRegionSettings(settings);
	applyYouTubeShields(settings);

	const queue = new TaskQueue({ concurrency: QUEUE_CONCURRENCY });
	const videos: VideoManager = createVideoManager();
	await videos.start();
	let urlByElement = new WeakMap<Element, string | null>();
	let pendingKeyByElement = new WeakMap<Element, string>();
	let discovery: DiscoveryHandle | null = null;
	/** Bumped on stop(): late verdicts/errors from before the stop are dropped. */
	let generation = 0;
	let active = false;

	const siteEnabled = () =>
		settings.status && !isAllowlisted(location.hostname, settings.allowlist);
	const siteEnabledFor = (s: Settings) =>
		s.status && !isAllowlisted(location.hostname, s.allowlist);
	const kindEnabled = (kind: MediaKind, s: Settings) =>
		kind === "video" ? s.blurVideos : s.blurImages;

	function applyVerdict(el: HTMLElement, verdict: Verdict, kind: MediaKind): void {
		if (el instanceof HTMLVideoElement) {
			// Once the sampler has verdicts of its own it owns the element's
			// state; a late poster verdict must not override it.
			const sampler = videos.samplerFor(el);
			if (sampler != null && sampler.decisions.length > 0) return;
		}
		setReasons(el, reasonsToJson(verdict.reasons));
		setState(el, verdict.decision);
		if (verdict.decision === "blur") {
			// A user-initiated reveal wins over a late-arriving verdict.
			if (el.dataset.hbReveal === "click") {
				setBlurred(el, false);
				removeRegionBlur(el);
			} else {
				applyBlurMode(el, verdict.reasons, kind, settings);
			}
		} else {
			setBlurred(el, false);
			removeRegionBlur(el);
		}
	}

	function applyError(el: HTMLElement): void {
		if (el instanceof HTMLVideoElement) {
			const sampler = videos.samplerFor(el);
			if (sampler != null && sampler.decisions.length > 0) return;
		}
		setState(el, "error");
		// Fail closed: an unreachable/failed engine keeps the blur only when
		// the user enabled failClosed (docs/PLAN.md goal 1).
		setBlurred(el, blurOnError(settings));
		removeRegionBlur(el);
	}

	async function classify(candidate: MediaCandidate, nonce?: string): Promise<Verdict> {
		const el = candidate.element;
		let url = candidate.url;
		if (url == null) throw new Error("candidate has no URL");
		// `recheckMedia` re-classifies: a fragment busts the engine's LRU cache
		// key without changing what the fetch resolves to.
		if (
			nonce != null &&
			!url.startsWith("data:") &&
			!url.startsWith("blob:") &&
			!url.includes("#hb-recheck")
		) {
			url = `${url}#hb-recheck-${nonce}`;
		}
		let frameUrl: string | undefined;
		const needsPixels =
			url.startsWith("data:") ||
			url.startsWith("blob:") ||
			(candidate.kind === "video" && el instanceof HTMLVideoElement && el.poster.length === 0);
		if (needsPixels && (el instanceof HTMLImageElement || el instanceof HTMLVideoElement)) {
			frameUrl = (await captureMediaFrame(el)) ?? undefined;
		}
		return classifyImage({ url, width: candidate.width, height: candidate.height, frameUrl });
	}

	function onCandidate(candidate: MediaCandidate, nonce?: string): void {
		const el = candidate.element;
		const isVideo = el instanceof HTMLVideoElement;
		setKind(el, candidate.kind);
		if (!kindEnabled(candidate.kind, settings)) {
			if (isVideo) videos.unobserve(el);
			clearAll(el);
			return;
		}
		const prevUrl = urlByElement.get(el);
		if (prevUrl === candidate.url) return;
		urlByElement.set(el, candidate.url);
		// Pending blur follows blurryStartMode only — failClosed governs
		// errors, not the loading state. Tiny icons/emojis skip the pulse.
		setState(el, "pending");
		setBlurred(el, blurWhilePending(settings));
		if (
			candidate.width > 0 &&
			candidate.height > 0 &&
			candidate.width <= TINY_ICON_MAX_PX &&
			candidate.height <= TINY_ICON_MAX_PX
		) {
			delete el.dataset.hbLoading;
		}
		// A re-classified element drops its stale region overlay — the pending
		// whole-element blur takes over until the new verdict lands.
		removeRegionBlur(el);
		// The video sampler owns frame classification for <video> elements;
		// the poster/src URL still gets the one-shot image classification —
		// unless the video already has frame data and is playing, in which
		// case the sampler covers it and a poster classify would be duplicate.
		if (isVideo) videos.observe(el);
		const samplerOwnsVideo =
			isVideo &&
			el instanceof HTMLVideoElement &&
			el.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
			!el.paused;
		if (candidate.url == null || samplerOwnsVideo) return; // a URL-bearing mutation re-emits it
		const key = `media:${candidate.url}`;
		pendingKeyByElement.set(el, key);
		const gen = generation;
		queue
			.enqueue(key, () => classify(candidate, nonce), candidate.visible ? "visible" : "background")
			.then((verdict) => {
				pendingKeyByElement.delete(el);
				if (gen === generation && active) applyVerdict(el, verdict, candidate.kind);
			})
			.catch(() => {
				pendingKeyByElement.delete(el);
				if (gen === generation && active) applyError(el);
			});
	}

	function onRemoved(el: HTMLElement): void {
		if (el.dataset.hbKind == null) return; // never a candidate
		if (el instanceof HTMLVideoElement) videos.unobserve(el);
		removeRegionBlur(el);
		clearAll(el);
	}

	function onVisible(el: HTMLElement): void {
		const key = pendingKeyByElement.get(el);
		if (key != null) queue.prioritize(key);
		// Layout may have drifted while the element was off-screen.
		syncRegionBlur(el);
	}

	function start(): void {
		active = true;
		void videos.start();
		discovery ??= startDiscovery(document, { onCandidate, onVisible, onRemoved });
	}

	function stop(): void {
		active = false;
		generation += 1;
		discovery?.stop();
		discovery = null;
		videos.dispose();
		queue.clear();
		clearAllRegionBlurs();
		urlByElement = new WeakMap();
		pendingKeyByElement = new WeakMap();
		for (const el of deepQueryAll<HTMLElement>(document, "[data-hb-kind]")) {
			removeRegionBlur(el);
			clearAll(el);
		}
	}

	/**
	 * Re-run policy over stored detections after a settings change. Elements
	 * skipped while their kind was disabled were never classified — they only
	 * carry `data-hb-kind`, so a kind flip false→true re-classifies them here
	 * instead of leaving them marked-but-blurred with no reasons.
	 */
	function redecideAll(next: Settings): void {
		for (const el of deepQueryAll<HTMLElement>(document, "[data-hb-kind]")) {
			const kind = el.dataset.hbKind as MediaKind;
			// The video sampler owns <video> elements; its own settings watch
			// (VideoManager.applySettings) re-evaluates them.
			if (kind === "video") continue;
			if (!kindEnabled(kind, next)) {
				setState(el, "clear");
				setBlurred(el, false);
				removeRegionBlur(el);
				continue;
			}
			const reasons = el.dataset.hbReasons;
			if (reasons != null) {
				const detections = reasonsFromJson(reasons);
				const decision = decide(detections, next);
				setState(el, decision);
				if (decision === "blur") {
					// A user-initiated reveal wins over a late-arriving verdict.
					if (el.dataset.hbReveal === "click") {
						setBlurred(el, false);
						removeRegionBlur(el);
					} else {
						applyBlurMode(el, detections, kind, next);
					}
				} else {
					setBlurred(el, false);
					removeRegionBlur(el);
				}
			} else if (el.dataset.hbState == null || el.dataset.hbState === "clear") {
				// Never classified (kind was off, or skipped before any verdict):
				// nothing stored to re-decide — classify it now if a URL exists.
				const candidate = candidateFor(el, el.dataset.hbVisible === "1");
				if (candidate != null && candidate.url != null) {
					urlByElement.delete(el);
					onCandidate(candidate);
				}
			} else {
				// Pending or erroring: the blur follows the current fail mode.
				setBlurred(el, el.dataset.hbState === "error" ? blurOnError(next) : blurWhilePending(next));
				removeRegionBlur(el);
			}
		}
	}

	// Reveal interactions, per the kind's revealMode. composedPath is
	// required: ev.target is retargeted to the shadow host for listeners
	// outside the shadow tree, so closest() would miss media inside shadow DOM.
	const mediaTarget = (ev: Event): HTMLElement | null => {
		const el = ev
			.composedPath()
			.find(
				(n): n is Element => n instanceof Element && n.matches("[data-hb-blur],[data-hb-region]"),
			);
		return el instanceof HTMLElement ? el : null;
	};
	const kindOf = (el: HTMLElement) => el.dataset.hbKind as MediaKind | undefined;

	// Permanent click reveal exists only in "hover" mode (the 0.2.x behavior
	// the boolean toggles used to gate); hover reveal itself is per-mode CSS
	// (whole-element) or the portal's JS timer (region mode).
	let swallowClickFor: HTMLElement | null = null;
	document.addEventListener(
		"click",
		(ev) => {
			const el = mediaTarget(ev);
			// A completed hold-to-peek swallows the click it leaves behind —
			// otherwise releasing a peek on a video would toggle playback of
			// the censored content.
			if (swallowClickFor != null) {
				const swallowing = el != null && el === swallowClickFor;
				swallowClickFor = null;
				if (swallowing) {
					ev.preventDefault();
					ev.stopPropagation();
					return;
				}
			}
			if (!(el instanceof HTMLElement)) return;
			const kind = kindOf(el);
			if (kind == null || revealModeFor(kind, settings) !== "hover") return;
			// A video's blur lives on its inline filter too — the manager must
			// clear it, not just the data attribute.
			if (el instanceof HTMLVideoElement) videos.reveal(el);
			else {
				reveal(el);
				removeRegionBlur(el);
			}
		},
		true,
	);

	// Hold-to-peek ("hold" mode): pressing a censored element for
	// HOLD_PEEK_DELAY_MS lifts the censor while the pointer stays down;
	// releasing anywhere re-blurs instantly. No permanent reveal.
	let peekEl: HTMLElement | null = null;
	let peekTimer: ReturnType<typeof setTimeout> | null = null;
	const endPeek = () => {
		if (peekTimer != null) {
			clearTimeout(peekTimer);
			peekTimer = null;
		}
		if (peekEl != null) {
			if (peekEl.dataset.hbPeek != null) swallowClickFor = peekEl;
			setPeeking(peekEl, false);
			peekRegionBlur(peekEl, false);
			peekEl = null;
		}
	};
	document.addEventListener(
		"pointerdown",
		(ev) => {
			endPeek();
			// A flag left over from an interrupted peek (no click followed)
			// must not swallow this press's click.
			swallowClickFor = null;
			const el = mediaTarget(ev);
			if (!(el instanceof HTMLElement)) return;
			const kind = kindOf(el);
			if (kind == null || revealModeFor(kind, settings) !== "hold") return;
			peekEl = el;
			peekTimer = setTimeout(() => {
				peekTimer = null;
				if (peekEl === el) {
					setPeeking(el, true);
					peekRegionBlur(el, true);
				}
			}, HOLD_PEEK_DELAY_MS);
		},
		true,
	);
	document.addEventListener("pointerup", endPeek, true);
	document.addEventListener("pointercancel", endPeek, true);

	/**
	 * Find the pipeline-tracked element showing `srcUrl` (context-menu actions
	 * arrive with the media's resolved URL).
	 */
	function findMediaBySrc(srcUrl: string): HTMLElement | null {
		for (const el of deepQueryAll<HTMLElement>(document, "[data-hb-kind]")) {
			const candidate = candidateFor(el, el.dataset.hbVisible === "1");
			if (candidate?.url === srcUrl) return el;
		}
		return null;
	}

	// Context-menu actions forwarded by the background via {tabId}
	// (docs/PLAN.md "Message protocol").
	onMessage("revealMedia", ({ data }) => {
		const el = findMediaBySrc(data.srcUrl);
		if (el == null) return;
		// "off" means no reveal paths at all — the context menu included.
		// In "hold" mode this stays the deliberate, high-friction escape.
		const kind = el.dataset.hbKind as MediaKind | undefined;
		if (kind == null || revealModeFor(kind, settings) === "off") return;
		if (el instanceof HTMLVideoElement) videos.reveal(el);
		else {
			reveal(el);
			removeRegionBlur(el);
		}
	});
	onMessage("recheckMedia", ({ data }) => {
		const el = findMediaBySrc(data.srcUrl);
		if (el == null) return;
		// Drop the dedupe entries so the classification actually re-runs, and
		// clear a previous click-reveal so a blur verdict re-applies.
		urlByElement.delete(el);
		pendingKeyByElement.delete(el);
		queue.delete(`media:${data.srcUrl}`);
		delete el.dataset.hbReveal;
		removeRegionBlur(el);
		const candidate = candidateFor(el, el.dataset.hbVisible === "1");
		if (candidate != null) onCandidate(candidate, String(Date.now()));
	});

	watchSettings((next, prev) => {
		const wasOn = siteEnabledFor(prev);
		const nowOn = siteEnabledFor(next);
		settings = next;
		applyBlurStyles(next);
		setRegionSettings(next);
		applyYouTubeShields(next);
		// A kind toggle false→true must re-scan: elements skipped while their
		// kind was disabled carry no data-hb-* marks for redecideAll to find.
		const kindEnabledFlip =
			(prev.blurImages === false && next.blurImages === true) ||
			(prev.blurVideos === false && next.blurVideos === true);
		if (nowOn === wasOn) {
			if (nowOn) {
				if (kindEnabledFlip) {
					stop();
					start();
				} else {
					redecideAll(next);
				}
			}
		} else if (nowOn) {
			start(); // re-scan: initial DOM pass re-emits everything
		} else {
			stop();
		}
	});

	if (siteEnabled()) start();
}
