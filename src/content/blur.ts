import type { RevealMode, Settings } from "../shared/settings";
import type { MediaKind } from "./discovery";

/**
 * Blur styling and element state (docs/PLAN.md "Blur", WP2).
 *
 * One injected `<style>` blurs every element carrying `data-hb-blur` with
 * `filter: blur(var(--hb-blur)) grayscale(var(--hb-gray))`. Element state lives
 * in `data-hb-*` attributes (state machine stays an attribute — cheap on feeds
 * with thousands of images):
 *
 * - `data-hb-kind`     "image" | "video" | "background"
 * - `data-hb-state`    "pending" | "blur" | "clear" | "error"
 * - `data-hb-blur`     present while the element is blurred
 * - `data-hb-loading`  present while a verdict is outstanding (loading state)
 * - `data-hb-visible`  set by the IntersectionObserver
 * - `data-hb-reasons`  `[{label, score}]` JSON from the verdict
 * - `data-hb-reveal`   "click" once the user revealed the element
 *
 * Fail closed: media is blurred until a verdict arrives; an error keeps the
 * blur unless `settings.failClosed` is false.
 */

const STYLE_ID = "hb-blur-styles";

/** Hover delay before a reveal, matching the 0.2.x extension's transition. */
const REVEAL_DELAY_MS = 1000;

/** Continuous hold time before a hold-to-peek reveal kicks in. */
export const HOLD_PEEK_DELAY_MS = 400;

export type MediaState = "pending" | "blur" | "clear" | "error";

/**
 * The document stylesheet can't cross shadow boundaries — media inside open
 * shadow roots (YouTube's `yt-img-shadow` thumbnails) needs the same rules
 * adopted into each shadow root. One shared CSSStyleSheet is appended to
 * `adoptedStyleSheets` (no DOM mutation inside the site's tree); custom
 * properties still inherit in from the document `:root`.
 */
let sharedSheet: CSSStyleSheet | null = null;
let lastCss = "";
const styledRoots = new WeakSet<ShadowRoot>();

function sharedBlurSheet(): CSSStyleSheet | null {
	if (typeof CSSStyleSheet === "undefined") return null;
	if (sharedSheet == null) {
		try {
			sharedSheet = new CSSStyleSheet();
			if (lastCss.length > 0) sharedSheet.replaceSync(lastCss);
		} catch {
			return null;
		}
	}
	return sharedSheet;
}

/** Adopt the blur stylesheet into the shadow root containing `el`, if any. */
export function ensureShadowStyles(el: HTMLElement): void {
	const root = el.getRootNode?.();
	if (root == null || typeof ShadowRoot === "undefined" || !(root instanceof ShadowRoot)) {
		return;
	}
	const sheet = sharedBlurSheet();
	if (sheet == null || styledRoots.has(root)) return;
	styledRoots.add(root);
	try {
		root.adoptedStyleSheets = [...root.adoptedStyleSheets, sheet];
	} catch {
		// Older engines without adoptedStyleSheets — region portal still works.
	}
}

/** Inject (or reuse) the blur stylesheet. Safe at document_start: attaches to
 * `documentElement` when `<head>` does not exist yet. */
export function ensureBlurStyles(doc: Document = document): HTMLStyleElement {
	let style = doc.getElementById(STYLE_ID) as HTMLStyleElement | null;
	if (style == null) {
		style = doc.createElement("style");
		style.id = STYLE_ID;
		(doc.head ?? doc.documentElement).appendChild(style);
	}
	return style;
}

/**
 * The interactive reveal mode for a media kind. Callers compare against
 * "hover"/"hold"/"off" — see the `RevealMode` docs in shared/settings.
 */
export function revealModeFor(kind: MediaKind, settings: Settings): RevealMode {
	return kind === "video" ? settings.revealModeVideos : settings.revealModeImages;
}

/**
 * Pending elements (verdict outstanding) stay blurred only in
 * `blurryStartMode` — the "start blurry" opt-in. Errors/timeouts are governed
 * separately by `failClosed` (docs/PLAN.md goal 1).
 */
export function blurWhilePending(settings: Settings): boolean {
	return settings.blurryStartMode;
}

/** Whether an element that failed classification stays blurred. */
export function blurOnError(settings: Settings): boolean {
	return settings.failClosed;
}

/** Regenerate the stylesheet for the current settings. */
export function applyBlurStyles(settings: Settings): void {
	const style = ensureBlurStyles();
	const blurPx = `${Math.round(settings.blurAmount)}px`;
	const gray = settings.gray ? "100%" : "0%";
	let css = `
:root {
	--hb-blur: ${blurPx};
	--hb-gray: ${gray};
	--hb-reveal-delay: ${REVEAL_DELAY_MS}ms;
}
[data-hb-blur] {
	filter: blur(var(--hb-blur)) grayscale(var(--hb-gray)) !important;
	transition: filter 0.1s ease !important;
}
@keyframes hb-loading {
	0%, 100% { filter: blur(var(--hb-blur)) grayscale(var(--hb-gray)); }
	50% { filter: blur(calc(var(--hb-blur) * 0.7)) grayscale(var(--hb-gray)); }
}
[data-hb-blur][data-hb-loading] {
	animation: hb-loading 1.4s ease-in-out infinite !important;
}`;
	// Hover reveal only for kinds the user allows, and only ever applied to
	// elements whose state isn't "blur-disabled" — see index.ts click gate.
	const hoverSelectors: string[] = [];
	if (settings.revealModeImages === "hover") {
		hoverSelectors.push('[data-hb-blur][data-hb-kind="image"]:hover');
		hoverSelectors.push('[data-hb-blur][data-hb-kind="background"]:hover');
	}
	if (settings.revealModeVideos === "hover") {
		hoverSelectors.push('[data-hb-blur][data-hb-kind="video"]:hover');
	}
	if (hoverSelectors.length > 0) {
		css += `
${hoverSelectors.join(",\n")} {
	filter: blur(0px) grayscale(0%) !important;
	transition: filter 0.45s ease !important;
	transition-delay: var(--hb-reveal-delay) !important;
	animation: none !important;
}`;
	}
	// Hold-to-peek: index.ts sets data-hb-peek while the pointer stays down.
	// Always emitted — the attribute only exists in "hold" mode anyway.
	css += `
[data-hb-blur][data-hb-peek] {
	filter: blur(0px) grayscale(0%) !important;
	transition: filter 0.2s ease !important;
	animation: none !important;
}`;
	style.textContent = css;
	lastCss = css;
	try {
		sharedSheet?.replaceSync(css);
	} catch {
		// Sheet not yet created or engine rejected the text — next
		// ensureShadowStyles call picks up lastCss.
	}
}

export function setKind(el: HTMLElement, kind: MediaKind): void {
	el.dataset.hbKind = kind;
	ensureShadowStyles(el);
}

export function setState(el: HTMLElement, state: MediaState): void {
	el.dataset.hbState = state;
	if (state === "pending") el.dataset.hbLoading = "1";
	else delete el.dataset.hbLoading;
}

export function setBlurred(el: HTMLElement, on: boolean): void {
	if (on) el.dataset.hbBlur = "1";
	else delete el.dataset.hbBlur;
}

export function setReasons(el: HTMLElement, reasonsJson: string): void {
	el.dataset.hbReasons = reasonsJson;
}

/** Mark the element peeked (hold-to-peek); the blur lifts while present. */
export function setPeeking(el: HTMLElement, on: boolean): void {
	if (on) el.dataset.hbPeek = "1";
	else delete el.dataset.hbPeek;
}

/** Mark the element revealed by click; clears the blur permanently. */
export function reveal(el: HTMLElement): void {
	setBlurred(el, false);
	el.dataset.hbReveal = "click";
}

/** Remove every `data-hb-*` attribute the pipeline set. */
export function clearAll(el: HTMLElement): void {
	for (const key of [
		"hbKind",
		"hbState",
		"hbBlur",
		"hbLoading",
		"hbVisible",
		"hbReasons",
		"hbReveal",
		"hbRegion",
		"hbPeek",
	]) {
		delete el.dataset[key];
	}
}
