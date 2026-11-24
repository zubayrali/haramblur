import { blurCapableDetections, type PolicySettings } from "../shared/policy";
import type { Detection } from "../shared/protocol";
import type { Settings } from "../shared/settings";
import { setBlurred } from "./blur";
import type { MediaKind } from "./discovery";

/**
 * Region blur (PLAN.md v1.1, pulled forward): instead of CSS-blurring the
 * whole element, an overlay sibling covering the element's box hosts one
 * `backdrop-filter` div per detected region. The media stays sharp outside
 * the boxes and blurred inside them — the visual parity the 0.2.x
 * whole-element blur never had.
 *
 * Mechanics:
 * - The overlay is an `position: absolute` next sibling. Siblings share the
 *   same offsetParent, so `offsetLeft/Top/Width/Height` align it to the
 *   element without any scroll tracking (absolute positions ride along with
 *   document scroll automatically).
 * - Region children use percentages of the overlay box, so they follow
 *   resizes; a shared ResizeObserver re-syncs the overlay box, and the
 *   content rect is recomputed each sync so `object-fit: cover` crops stay
 *   aligned.
 * - `backdrop-filter` samples whatever is painted beneath — works for <img>
 *   today; <video> could reuse it later (the sampler owns video blur state).
 * - `pointer-events: none` keeps hover/click reveal working through the
 *   existing handlers; the hover-reveal CSS targets the overlay via the
 *   adjacent-sibling selector.
 */

const OVERLAY_CLASS = "hb-region-overlay";
const REGION_CLASS = "hb-region";
const MAX_REGIONS = 16;
/**
 * Inflate each box so the blur's soft edge never leaves a sharp sliver of
 * the detected region visible. In normalized image units.
 */
const REGION_PAD = 0.04;
/** Max allowed drift between element and overlay rects before we bail to
 * whole-element blur — catches offsetParent edge cases (filters, tables…). */
const ALIGN_TOLERANCE_PX = 2;

export interface PixelRect {
	x: number;
	y: number;
	w: number;
	h: number;
}

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/** Grow a normalized box by `pad` on every side, clamped to 0..1. */
export function padBox(box: Detection["box"], pad = REGION_PAD): Detection["box"] {
	const [x, y, w, h] = box;
	const x0 = Math.max(0, x - pad);
	const y0 = Math.max(0, y - pad);
	const x1 = Math.min(1, x + w + pad);
	const y1 = Math.min(1, y + h + pad);
	// Round to 4 decimals — sub-pixel precision isn't needed and it keeps the
	// serialized boxes (and tests) free of float noise.
	const r = (v: number) => Math.round(v * 1e4) / 1e4;
	return [r(x0), r(y0), r(x1 - x0), r(y1 - y0)];
}

/**
 * Normalized boxes of every detection that meets the blur bar — exactly the
 * regions that would have triggered the whole-element blur, highest score
 * first, padded and capped.
 */
export function blurRegionBoxes(
	detections: readonly Detection[],
	settings: PolicySettings,
): Detection["box"][] {
	return blurCapableDetections(detections, settings)
		.sort((a, b) => b.score - a.score)
		.slice(0, MAX_REGIONS)
		.map((d) => padBox(d.box));
}

/**
 * Parse an `object-position` value into [x, y] fractions (0 = left/top edge,
 * 0.5 = center, 1 = right/bottom). Handles keywords and percentages — the
 * forms feed CSS actually uses; lengths and the 3/4-value offset syntax
 * degrade to center, which is the safe default.
 */
export function parseObjectPosition(position: string): [number, number] {
	const tokens = position.trim().toLowerCase().split(/\s+/).filter(Boolean);
	const axis = (token: string | undefined, isX: boolean): number => {
		if (token == null || token === "") return 0.5;
		if (isX ? token === "left" : token === "top") return 0;
		if (isX ? token === "right" : token === "bottom") return 1;
		if (token === "center") return 0.5;
		const m = /^(-?\d+(?:\.\d+)?)%$/.exec(token);
		if (m != null) return clamp01(Number.parseFloat(m[1]!) / 100);
		return 0.5;
	};
	// Single-value syntax centers the other axis — except a lone vertical
	// keyword ("top"/"bottom"), which still sets y.
	const yToken =
		tokens.length >= 2
			? tokens[1]
			: tokens[0] === "top" || tokens[0] === "bottom"
				? tokens[0]
				: undefined;
	return [axis(tokens[0], true), axis(yToken, false)];
}

/**
 * The rectangle (in element-box pixels) that the image's pixels actually
 * occupy under `object-fit`/`object-position`. `fill` (the default) covers
 * the whole element box; `cover` crops, `contain` letterboxes, `none` and
 * `scale-down` place the natural-size image at the position anchor.
 */
export function imageContentRect(
	ew: number,
	eh: number,
	nw: number,
	nh: number,
	fit: string,
	position: string,
): PixelRect {
	if (nw <= 0 || nh <= 0 || ew <= 0 || eh <= 0) return { x: 0, y: 0, w: ew, h: eh };
	if (fit === "fill" || fit === "") return { x: 0, y: 0, w: ew, h: eh };
	const cw = ew / nw;
	const ch = eh / nh;
	let scale: number;
	switch (fit) {
		case "contain":
			scale = Math.min(cw, ch);
			break;
		case "cover":
			scale = Math.max(cw, ch);
			break;
		case "none":
			scale = 1;
			break;
		case "scale-down":
			scale = Math.min(1, Math.min(cw, ch));
			break;
		default:
			return { x: 0, y: 0, w: ew, h: eh };
	}
	const w = nw * scale;
	const h = nh * scale;
	const [px, py] = parseObjectPosition(position);
	// x = px·(ew − w) works for cover (negative → crop), contain (positive →
	// letterbox) and none alike. `+ 0` normalizes -0 from 0·negative products.
	return { x: px * (ew - w) + 0, y: py * (eh - h) + 0, w, h };
}

/** Map a normalized image-space box into element-box pixels via the content rect. */
export function regionToElementPx(box: Detection["box"], content: PixelRect): PixelRect {
	const [nx, ny, nw, nh] = box;
	return {
		x: content.x + nx * content.w,
		y: content.y + ny * content.h,
		w: nw * content.w,
		h: nh * content.h,
	};
}

// ---------------------------------------------------------------------------
// DOM overlay management
// ---------------------------------------------------------------------------

const overlays = new WeakMap<HTMLElement, HTMLElement>();
const tracked = new Set<HTMLElement>();
let observer: ResizeObserver | null = null;

function ensureObserver(): ResizeObserver {
	observer ??= new ResizeObserver((entries) => {
		for (const entry of entries) syncRegionBlur(entry.target as HTMLElement);
	});
	return observer;
}

/**
 * Region blur needs layout we can mirror with an absolutely-positioned
 * sibling: a loaded <img>, not fixed/sticky positioned, not transformed or
 * z-indexed (those desync offsetLeft from the painted box).
 */
export function canRegionBlur(el: HTMLElement): el is HTMLImageElement {
	if (typeof HTMLImageElement === "undefined" || !(el instanceof HTMLImageElement)) {
		return false;
	}
	if (el.naturalWidth === 0 || el.naturalHeight === 0) return false;
	const cs = getComputedStyle(el);
	if (cs.position === "fixed" || cs.position === "sticky") return false;
	if (cs.transform !== "none") return false;
	if (cs.zIndex !== "auto" && cs.zIndex !== "0") return false;
	return true;
}

function readBoxes(overlay: HTMLElement): Detection["box"][] {
	try {
		const parsed: unknown = JSON.parse(overlay.dataset.hbBoxes ?? "[]");
		return Array.isArray(parsed) ? (parsed as Detection["box"][]) : [];
	} catch {
		return [];
	}
}

/**
 * (Re)position the overlay over the element and rebuild region children.
 * Runs on create, ResizeObserver callbacks and manual resyncs.
 */
export function syncRegionBlur(el: HTMLElement): void {
	const overlay = overlays.get(el);
	if (overlay == null) return;
	if (!el.isConnected) {
		removeRegionBlur(el);
		return;
	}
	const ow = el.offsetWidth;
	const oh = el.offsetHeight;
	if (ow === 0 || oh === 0) {
		overlay.style.display = "none";
		return;
	}
	overlay.style.display = "";
	overlay.style.left = `${el.offsetLeft}px`;
	overlay.style.top = `${el.offsetTop}px`;
	overlay.style.width = `${ow}px`;
	overlay.style.height = `${oh}px`;

	const cs = getComputedStyle(el);
	// Only elements that passed canRegionBlur (HTMLImageElement) are tracked.
	const img = el as HTMLImageElement;
	const content = imageContentRect(
		ow,
		oh,
		img.naturalWidth,
		img.naturalHeight,
		cs.objectFit,
		cs.objectPosition,
	);
	const frag = document.createDocumentFragment();
	for (const box of readBoxes(overlay)) {
		const r = regionToElementPx(box, content);
		const region = document.createElement("div");
		region.className = REGION_CLASS;
		region.style.left = `${(r.x / ow) * 100}%`;
		region.style.top = `${(r.y / oh) * 100}%`;
		region.style.width = `${(r.w / ow) * 100}%`;
		region.style.height = `${(r.h / oh) * 100}%`;
		frag.appendChild(region);
	}
	overlay.replaceChildren(frag);
}

/**
 * Show region blur for `el` covering `boxes` (normalized over image pixels).
 * Returns true when the overlay is live; on any layout mismatch it removes
 * itself, calls `onFallback` (whole-element blur) and returns false.
 */
export function applyRegionBlur(
	el: HTMLElement,
	boxes: readonly Detection["box"][],
	onFallback?: () => void,
): boolean {
	removeRegionBlur(el);
	if (boxes.length === 0 || !canRegionBlur(el)) return false;
	const overlay = document.createElement("div");
	overlay.className = OVERLAY_CLASS;
	overlay.dataset.hbBoxes = JSON.stringify(boxes);
	el.insertAdjacentElement("afterend", overlay);
	overlays.set(el, overlay);
	tracked.add(el);
	ensureObserver().observe(el);
	syncRegionBlur(el);
	// Verify the sibling actually landed on the element's painted box —
	// offsetParent edge cases (tables, filters on ancestors) can desync it.
	const a = el.getBoundingClientRect();
	const b = overlay.getBoundingClientRect();
	const aligned =
		Math.abs(a.left - b.left) <= ALIGN_TOLERANCE_PX &&
		Math.abs(a.top - b.top) <= ALIGN_TOLERANCE_PX &&
		Math.abs(a.width - b.width) <= ALIGN_TOLERANCE_PX &&
		Math.abs(a.height - b.height) <= ALIGN_TOLERANCE_PX;
	if (!aligned) {
		removeRegionBlur(el);
		onFallback?.();
		return false;
	}
	// Late reflow (images above finishing) can shift the element after the
	// verdict; resync once the frame settles and again after ~1 s.
	requestAnimationFrame(() => syncRegionBlur(el));
	setTimeout(() => syncRegionBlur(el), 1000);
	return true;
}

/** Tear down the overlay for `el`, if any. Idempotent. */
export function removeRegionBlur(el: HTMLElement): void {
	const overlay = overlays.get(el);
	if (overlay != null) {
		overlay.remove();
		overlays.delete(el);
	}
	tracked.delete(el);
	observer?.unobserve(el);
	delete el.dataset.hbRegion;
}

/** Whether `el` currently has a live region overlay. */
export function hasRegionBlur(el: HTMLElement): boolean {
	return overlays.has(el);
}

/** Re-sync every tracked overlay — used on viewport/scroll-adjacent events. */
export function syncAllRegionBlurs(): void {
	for (const el of tracked) syncRegionBlur(el);
}

/**
 * Apply the blur mode for a blur verdict: region overlay when eligible,
 * whole-element `data-hb-blur` otherwise. Centralizes the decision so
 * verdict application and settings re-decisions stay identical.
 */
export function applyBlurMode(
	el: HTMLElement,
	detections: readonly Detection[],
	kind: MediaKind,
	settings: Settings,
): void {
	removeRegionBlur(el);
	setBlurred(el, false);
	if (settings.regionBlur && kind === "image") {
		const boxes = blurRegionBoxes(detections, settings);
		if (
			boxes.length > 0 &&
			applyRegionBlur(el, boxes, () => {
				// Layout couldn't mirror the element — degrade to whole blur.
				setBlurred(el, true);
			})
		) {
			el.dataset.hbRegion = "1";
			return;
		}
	}
	setBlurred(el, true);
}
