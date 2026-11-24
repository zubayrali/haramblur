import { onMessage, type Verdict } from "../shared/protocol";
import { getSettings, type Settings, watchSettings } from "../shared/settings-store";
import {
	applyBlurStyles,
	blurOnUncertainty,
	clearAll,
	ensureBlurStyles,
	reveal,
	revealAllowed,
	setBlurred,
	setKind,
	setReasons,
	setState,
} from "./blur";
import {
	candidateFor,
	type DiscoveryHandle,
	type MediaCandidate,
	type MediaKind,
	startDiscovery,
} from "./discovery";
import { classifyImage } from "./engine";
import { captureMediaFrame } from "./frames";
import { TaskQueue } from "./queue";
import { isAllowlisted, reasonsToJson, redecide } from "./site";
import { createVideoManager, type VideoManager } from "./video";

/**
 * Content pipeline glue (docs/PLAN.md WP2). Deliberately small: discovery emits
 * candidates, the queue classifies them through the engine client, and blur.ts
 * owns the styling. No engine imports — only the `src/shared/` protocol.
 */

const QUEUE_CONCURRENCY = 4;

export async function startContentPipeline(): Promise<void> {
	let settings = await getSettings();
	ensureBlurStyles();
	applyBlurStyles(settings);

	const queue = new TaskQueue({ concurrency: QUEUE_CONCURRENCY });
	const videos: VideoManager = createVideoManager();
	await videos.start();
	let urlByElement = new WeakMap<Element, string | null>();
	let pendingKeyByElement = new WeakMap<Element, string>();
	let discovery: DiscoveryHandle | null = null;

	const siteEnabled = () =>
		settings.status && !isAllowlisted(location.hostname, settings.allowlist);
	const siteEnabledFor = (s: Settings) =>
		s.status && !isAllowlisted(location.hostname, s.allowlist);
	const kindEnabled = (kind: MediaKind, s: Settings) =>
		kind === "video" ? s.blurVideos : s.blurImages;

	function applyVerdict(el: HTMLElement, verdict: Verdict): void {
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
			if (el.dataset.hbReveal !== "click") setBlurred(el, true);
		} else {
			setBlurred(el, false);
		}
	}

	function applyError(el: HTMLElement): void {
		if (el instanceof HTMLVideoElement) {
			const sampler = videos.samplerFor(el);
			if (sampler != null && sampler.decisions.length > 0) return;
		}
		setState(el, "error");
		// Fail closed: an unreachable/failed engine keeps the blur unless the
		// user disabled it (docs/PLAN.md goal 1).
		setBlurred(el, blurOnUncertainty(settings));
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
		// Fail closed: blur while the verdict is outstanding.
		setState(el, "pending");
		setBlurred(el, blurOnUncertainty(settings));
		// The video sampler owns frame classification for <video> elements;
		// the poster/src URL still gets the one-shot image classification.
		if (isVideo) videos.observe(el);
		if (candidate.url == null) return; // a URL-bearing mutation re-emits it
		const key = `media:${candidate.url}`;
		pendingKeyByElement.set(el, key);
		queue
			.enqueue(key, () => classify(candidate, nonce), candidate.visible ? "visible" : "background")
			.then((verdict) => {
				pendingKeyByElement.delete(el);
				applyVerdict(el, verdict);
			})
			.catch(() => {
				pendingKeyByElement.delete(el);
				applyError(el);
			});
	}

	function onRemoved(el: HTMLElement): void {
		if (el.dataset.hbKind == null) return; // never a candidate
		if (el instanceof HTMLVideoElement) videos.unobserve(el);
		clearAll(el);
	}

	function onVisible(el: HTMLElement): void {
		const key = pendingKeyByElement.get(el);
		if (key != null) queue.prioritize(key);
	}

	function start(): void {
		discovery ??= startDiscovery(document, { onCandidate, onVisible, onRemoved });
	}

	function stop(): void {
		discovery?.stop();
		discovery = null;
		videos.dispose();
		urlByElement = new WeakMap();
		pendingKeyByElement = new WeakMap();
		for (const el of document.querySelectorAll<HTMLElement>("[data-hb-kind]")) {
			clearAll(el);
		}
	}

	/** Re-run policy over stored detections after a settings change. */
	function redecideAll(next: Settings): void {
		for (const el of document.querySelectorAll<HTMLElement>("[data-hb-kind]")) {
			const kind = el.dataset.hbKind as MediaKind;
			// The video sampler owns <video> elements; its own settings watch
			// (VideoManager.applySettings) re-evaluates them.
			if (kind === "video") continue;
			if (!kindEnabled(kind, next)) {
				setState(el, "clear");
				setBlurred(el, false);
				continue;
			}
			const reasons = el.dataset.hbReasons;
			if (reasons != null) {
				const decision = redecide(reasons, next);
				setState(el, decision);
				if (decision === "blur") {
					if (el.dataset.hbReveal !== "click") setBlurred(el, true);
				} else {
					setBlurred(el, false);
				}
			} else {
				// Pending or erroring: the blur follows the current fail mode.
				setBlurred(el, blurOnUncertainty(next));
			}
		}
	}

	// Reveal on click where settings allow; hover reveal is pure CSS.
	document.addEventListener(
		"click",
		(ev) => {
			const el = (ev.target as Element | null)?.closest?.("[data-hb-blur]");
			if (!(el instanceof HTMLElement)) return;
			const kind = el.dataset.hbKind as MediaKind | undefined;
			if (kind == null || !revealAllowed(kind, settings)) return;
			// A video's blur lives on its inline filter too — the manager must
			// clear it, not just the data attribute.
			if (el instanceof HTMLVideoElement) videos.reveal(el);
			else reveal(el);
		},
		true,
	);

	/**
	 * Find the pipeline-tracked element showing `srcUrl` (context-menu actions
	 * arrive with the media's resolved URL).
	 */
	function findMediaBySrc(srcUrl: string): HTMLElement | null {
		for (const el of document.querySelectorAll<HTMLElement>("[data-hb-kind]")) {
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
		if (el instanceof HTMLVideoElement) videos.reveal(el);
		else reveal(el);
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
		const candidate = candidateFor(el, el.dataset.hbVisible === "1");
		if (candidate != null) onCandidate(candidate, String(Date.now()));
	});

	watchSettings((next, prev) => {
		const wasOn = siteEnabledFor(prev);
		const nowOn = siteEnabledFor(next);
		settings = next;
		applyBlurStyles(next);
		if (nowOn === wasOn) {
			if (nowOn) redecideAll(next);
		} else if (nowOn) {
			start(); // re-scan: initial DOM pass re-emits everything
		} else {
			stop();
		}
	});

	if (siteEnabled()) start();
}
