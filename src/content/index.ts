import type { Verdict } from "../shared/protocol";
import { getSettings, type Settings, watchSettings } from "../shared/settings";
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
	type DiscoveryHandle,
	type MediaCandidate,
	type MediaKind,
	startDiscovery,
} from "./discovery";
import { classifyImage } from "./engine";
import { captureMediaFrame } from "./frames";
import { TaskQueue } from "./queue";
import { isAllowlisted, reasonsToJson, redecide } from "./site";

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
		setState(el, "error");
		// Fail closed: an unreachable/failed engine keeps the blur unless the
		// user disabled it (docs/PLAN.md goal 1).
		setBlurred(el, blurOnUncertainty(settings));
	}

	async function classify(candidate: MediaCandidate): Promise<Verdict> {
		const el = candidate.element;
		const url = candidate.url;
		if (url == null) throw new Error("candidate has no URL");
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

	function onCandidate(candidate: MediaCandidate): void {
		const el = candidate.element;
		setKind(el, candidate.kind);
		if (!kindEnabled(candidate.kind, settings)) {
			clearAll(el);
			return;
		}
		const prevUrl = urlByElement.get(el);
		if (prevUrl === candidate.url) return;
		urlByElement.set(el, candidate.url);
		// Fail closed: blur while the verdict is outstanding.
		setState(el, "pending");
		setBlurred(el, blurOnUncertainty(settings));
		if (candidate.url == null) return; // a URL-bearing mutation re-emits it
		const key = `media:${candidate.url}`;
		pendingKeyByElement.set(el, key);
		queue
			.enqueue(key, () => classify(candidate), candidate.visible ? "visible" : "background")
			.then((verdict) => {
				pendingKeyByElement.delete(el);
				applyVerdict(el, verdict);
			})
			.catch(() => {
				pendingKeyByElement.delete(el);
				applyError(el);
			});
	}

	function onVisible(el: HTMLElement): void {
		const key = pendingKeyByElement.get(el);
		if (key != null) queue.prioritize(key);
	}

	function start(): void {
		discovery ??= startDiscovery(document, { onCandidate, onVisible });
	}

	function stop(): void {
		discovery?.stop();
		discovery = null;
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
			reveal(el);
		},
		true,
	);

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
