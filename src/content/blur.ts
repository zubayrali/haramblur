import type { Settings } from "../shared/settings";
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

export type MediaState = "pending" | "blur" | "clear" | "error";

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

/** Whether hover/click reveal is enabled for a media kind. */
export function revealAllowed(kind: MediaKind, settings: Settings): boolean {
	return kind === "video" ? settings.unblurVideos : settings.unblurImages;
}

/** Whether a pending/erroring element stays blurred under this fail mode. */
export function blurOnUncertainty(settings: Settings): boolean {
	return settings.failClosed || settings.blurryStartMode;
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
	opacity: unset !important;
}
/* Region blur (region.ts): an absolutely-positioned sibling covers the
   element; each .hb-region backdrop-filters only its detection box. */
.hb-region-overlay {
	position: absolute;
	pointer-events: none;
	transition: opacity 0.1s ease;
}
.hb-region {
	position: absolute;
	backdrop-filter: blur(var(--hb-blur)) grayscale(var(--hb-gray));
	-webkit-backdrop-filter: blur(var(--hb-blur)) grayscale(var(--hb-gray));
	border-radius: 12%;
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
	if (settings.unblurImages) {
		hoverSelectors.push('[data-hb-blur][data-hb-kind="image"]:hover');
		hoverSelectors.push('[data-hb-blur][data-hb-kind="background"]:hover');
	}
	if (settings.unblurVideos) {
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
	// Region mode: fade the overlay out on hover (it's the next sibling).
	if (settings.unblurImages && settings.regionBlur) {
		css += `
[data-hb-region][data-hb-kind="image"]:hover + .hb-region-overlay {
	opacity: 0 !important;
	transition: opacity 0.45s ease !important;
	transition-delay: var(--hb-reveal-delay) !important;
}`;
	}
	style.textContent = css;
}

export function setKind(el: HTMLElement, kind: MediaKind): void {
	el.dataset.hbKind = kind;
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
	]) {
		delete el.dataset[key];
	}
}
