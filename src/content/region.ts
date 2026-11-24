import { blurCapableDetections, type PolicySettings } from "../shared/policy";
import type { Detection } from "../shared/protocol";
import type { Settings } from "../shared/settings";
import { revealAllowed, setBlurred } from "./blur";
import { extractBackgroundImageUrl, type MediaKind } from "./discovery";

/**
 * Region blur (PLAN.md v1.1) via a Shadow-DOM overlay portal.
 *
 * Instead of CSS-blurring the whole element, one fixed-position portal host
 * appended to `documentElement` carries a shadow root with a `backdrop-filter`
 * div per blur-capable detection box. The media stays sharp outside the boxes
 * and blurred inside them — and because nothing is inserted next to the media
 * element, React-managed parents (X.com) can't have their DOM mutated.
 *
 * Mechanics:
 * - One `<div id="hb-region-portal">` (position: fixed; inset: 0;
 *   pointer-events: none; z-index: max) holds an open shadow root. Each
 *   region-blurred element gets a `.hb-rh` host div positioned at its
 *   *clipped* viewport rect (element ∩ overflow/clip-path ancestors), with
 *   `overflow: hidden` so off-clip regions are cut exactly like the page.
 * - Region children are positioned in viewport pixels inside the host, so
 *   they follow scroll/resize via getBoundingClientRect — no offsetParent
 *   assumptions, works under transforms, filters and z-index.
 * - Sync triggers: ResizeObserver + IntersectionObserver per element,
 *   capture-phase passive scroll, window resize — all rAF-throttled. Every
 *   flush prunes disconnected elements.
 * - `object-fit`/`object-position` (img, video) and `background-size`/
 *   `background-position` (bg divs) are mapped through the same
 *   `imageContentRect` math so regions land on the painted pixels.
 * - Hover reveal fades the host's opacity after REVEAL_DELAY_MS, matching
 *   the whole-element CSS reveal; gated by `revealAllowed(kind, settings)`.
 * - Whole-element `data-hb-blur` remains the fallback for full-frame
 *   detections (e.g. NSFW_*), `regionBlur: false`, unloaded media, and
 *   background images whose natural size hasn't decoded yet (they upgrade
 *   to region blur once it arrives).
 */

// ---------------------------------------------------------------------------
// Pure geometry helpers (shared with tests)
// ---------------------------------------------------------------------------

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

const FACE_KIND_FOR_COVERAGE: ReadonlySet<string> = new Set(["FACE_FEMALE", "FACE_MALE"]);

/**
 * Expand a blur-capable detection to the user's `coverageMode` before the
 * uniform pad. Face boxes grow asymmetrically — hair sits above the
 * hairline and to the sides, the neck below the chin; `body` estimates a
 * head-and-torso silhouette (~2.4× face width, crown to mid-torso).
 * Non-face labels keep their detector box (body-part boxes are already
 * anatomical).
 */
export function coverageBoxFor(d: Detection, mode: Settings["coverageMode"]): Detection["box"] {
	if (mode === "face" || !FACE_KIND_FOR_COVERAGE.has(d.label)) return d.box;
	const [x, y, w, h] = d.box;
	const r = (v: number) => Math.round(v * 1e4) / 1e4;
	const clampBox = (bx: number, by: number, bw: number, bh: number): Detection["box"] => {
		const x0 = Math.min(1, Math.max(0, bx));
		const y0 = Math.min(1, Math.max(0, by));
		const x1 = Math.min(1, Math.max(0, bx + bw));
		const y1 = Math.min(1, Math.max(0, by + bh));
		return [r(x0), r(y0), r(Math.max(0, x1 - x0)), r(Math.max(0, y1 - y0))];
	};
	if (mode === "head") {
		// +0.55h above (hair/forehead), ±0.35w sides (ears/hair), +0.35h below (neck).
		return clampBox(x - 0.35 * w, y - 0.55 * h, w * 1.7, h * 1.9);
	}
	// body: shoulders ≈2.4× face width centered on the face, crown to
	// ~4.5 face-heights below the chin (head + torso).
	const bw = w * 2.4;
	return clampBox(x + w / 2 - bw / 2, y - 0.55 * h, bw, h * (1 + 0.55 + 4.5));
}

/**
 * Normalized boxes of every detection that meets the blur bar — exactly the
 * regions that would have triggered the whole-element blur, expanded to
 * `coverageMode`, highest score first, padded and capped.
 */
export function blurRegionBoxes(
	detections: readonly Detection[],
	settings: Settings,
): Detection["box"][] {
	return blurCapableDetections(detections, settings)
		.sort((a, b) => b.score - a.score)
		.slice(0, MAX_REGIONS)
		.map((d) => padBox(coverageBoxFor(d, settings.coverageMode)));
}

/** Approximate covered fraction of the frame — box areas summed, overlaps
 * overcount (conservative: pushes toward whole-censor, never away). */
export function boxesCoveredArea(boxes: readonly Detection["box"][]): number {
	let area = 0;
	for (const [, , w, h] of boxes) area += w * h;
	return Math.min(1, area);
}

/** Female faces among the blur-capable detections — the "subjects" signal. */
export function femaleFaceCount(
	detections: readonly Detection[],
	settings: PolicySettings,
): number {
	return blurCapableDetections(detections, settings).filter((d) => d.label === "FACE_FEMALE")
		.length;
}

/**
 * Decide whether a blur verdict paints regions or censors the whole element.
 * Whole-element wins when: region blur is off or `wholeImages` covers this
 * kind, any detection is full-frame, `smartCensor` sees explicit content /
 * ≥2 female subjects / region coverage over `escalateAreaPct`, or no
 * localizable boxes remain.
 */
export function planCensor(
	detections: readonly Detection[],
	kind: MediaKind,
	settings: Settings,
): { whole: boolean; boxes: Detection["box"][] } {
	const whole = { whole: true, boxes: [] as Detection["box"][] };
	if (!settings.regionBlur || (settings.wholeImages && kind !== "video")) return whole;
	const capable = blurCapableDetections(detections, settings);
	if (capable.length === 0 || capable.some((d) => isFullFrameBox(d.box))) return whole;
	const boxes = blurRegionBoxes(detections, settings);
	if (boxes.length === 0) return whole;
	if (settings.smartCensor) {
		if (capable.some((d) => WHOLE_CENSOR_LABELS.has(d.label))) return whole;
		if (femaleFaceCount(detections, settings) >= SMART_CENSOR_MAX_FEMALE_FACES) return whole;
		if (boxesCoveredArea(boxes) > settings.escalateAreaPct) return whole;
	}
	return { whole: false, boxes };
}

/** A box covering essentially the whole frame — nothing to localize. */
export function isFullFrameBox(box: Detection["box"]): boolean {
	const [x, y, w, h] = box;
	return x <= 0.02 && y <= 0.02 && w >= 0.96 && h >= 0.96;
}

/**
 * Parse an `object-position`/`background-position` value into [x, y]
 * fractions (0 = left/top edge, 0.5 = center, 1 = right/bottom edge).
 * Handles keywords and percentages — the forms feed CSS actually uses;
 * lengths and the 3/4-value offset syntax degrade to center.
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

/** First layer of a comma-separated CSS value (paren-aware). */
function firstLayer(value: string): string {
	let depth = 0;
	for (let i = 0; i < value.length; i++) {
		const ch = value[i];
		if (ch === "(") depth++;
		else if (ch === ")") depth--;
		else if (ch === "," && depth === 0) return value.slice(0, i).trim();
	}
	return value.trim();
}

/**
 * `background-size` → the `imageContentRect` fit vocabulary. `cover`/`contain`
 * map directly; `auto` means intrinsic size ("none"); explicit lengths and
 * percentages degrade to "fill" — they can't be expressed as a single fit,
 * and stretching is the closest sane approximation for region anchoring.
 */
export function backgroundFitFor(size: string): string {
	const v = firstLayer(size).toLowerCase();
	if (v === "cover" || v === "contain") return v;
	if (/^auto(\s|$)/.test(v)) return "none";
	return "fill";
}

const NATURAL_SIZE: Map<string, { w: number; h: number }> = new Map();
const NATURAL_PENDING = new Set<string>();

/**
 * Kick off an async decode of `url`'s natural size (deduped). On completion,
 * pending background entries re-run their region apply.
 */
function ensureNaturalSize(url: string): void {
	if (NATURAL_SIZE.has(url) || NATURAL_PENDING.has(url)) return;
	if (typeof Image !== "function") return; // no decoder outside a DOM
	NATURAL_PENDING.add(url);
	const img = new Image();
	img.onload = () => {
		NATURAL_PENDING.delete(url);
		NATURAL_SIZE.set(url, { w: img.naturalWidth, h: img.naturalHeight });
		for (const el of [...(upgradesByUrl.get(url) ?? [])]) fulfillUpgrade(el);
		scheduleSyncAll();
	};
	img.onerror = () => {
		NATURAL_PENDING.delete(url);
		// Leave uncached — a later apply retries the decode.
	};
	img.src = url;
}

/** Natural pixel size of a URL once decoded; null while unknown. */
export function naturalSizeFor(url: string): { w: number; h: number } | null {
	return NATURAL_SIZE.get(url) ?? null;
}

// ---------------------------------------------------------------------------
// Portal DOM
// ---------------------------------------------------------------------------

const PORTAL_ID = "hb-region-portal";
const HOST_CLASS = "hb-rh";
const REGION_CLASS = "hb-rr";
const MAX_REGIONS = 16;
/** Inflate each box so the blur's soft edge never leaves a sharp sliver. */
const REGION_PAD = 0.04;
/** `smartCensor`: this many blur-capable female faces → whole-element. */
const SMART_CENSOR_MAX_FEMALE_FACES = 2;
/** Normalized box covering the whole media — used for style-driven
 * whole-element censors that paint through the portal. */
const FULL_FRAME_BOX: Detection["box"] = [0, 0, 1, 1];
/**
 * Explicit classes that always escalate to whole-element under `smartCensor`.
 * Breast exposure deliberately stays regional — swimwear/suggestive content
 * is exactly what region blur exists for; genitalia/anus/buttocks mean the
 * rest of the frame is part of the same exposure.
 */
const WHOLE_CENSOR_LABELS: ReadonlySet<string> = new Set([
	"FEMALE_GENITALIA_EXPOSED",
	"MALE_GENITALIA_EXPOSED",
	"ANUS_EXPOSED",
	"BUTTOCKS_EXPOSED",
]);
/** Reveal delay matching the whole-element CSS hover reveal. */
const REVEAL_DELAY_MS = 1_000;
/** Offscreen culling margin (px beyond the viewport) for sync writes. */
const OFFSCREEN_MARGIN = 2_000;

const entries = new Map<HTMLElement, RegionEntry>();
const pendingSync = new Set<HTMLElement>();
let portal: HTMLElement | null = null;
let portalRoot: ShadowRoot | null = null;
let globalListenersBound = false;
let activeSettings: Settings | null = null;

interface RegionEntry {
	el: HTMLElement;
	kind: MediaKind;
	/** Padded, blur-capable normalized boxes (over image pixels). */
	boxes: Detection["box"][];
	/** Visual treatment for this entry's region children. */
	style: Settings["censorStyle"];
	/** Host div inside the shadow root, positioned at the clipped rect. */
	host: HTMLElement;
	/** Region children parallel to `boxes`. */
	regions: HTMLElement[];
	resizeObserver: ResizeObserver | null;
	intersectionObserver: IntersectionObserver | null;
	revealTimer: ReturnType<typeof setTimeout> | null;
	hoverReveal: boolean;
}

/** Settings source for hover-reveal gating; index.ts keeps it current. */
export function setRegionSettings(settings: Settings | null): void {
	activeSettings = settings;
}

function ensurePortal(): ShadowRoot {
	if (portalRoot != null) return portalRoot;
	portal = document.createElement("div");
	portal.id = PORTAL_ID;
	portal.style.cssText =
		"position:fixed;inset:0;pointer-events:none;z-index:2147483646;overflow:hidden;";
	portalRoot = portal.attachShadow({ mode: "open" });
	const style = document.createElement("style");
	style.textContent = `
.${HOST_CLASS} {
	position: absolute;
	overflow: hidden;
	pointer-events: none;
	transition: opacity 0.45s ease;
}
.${REGION_CLASS} {
	position: absolute;
	backdrop-filter: blur(var(--hb-blur, 20px)) grayscale(var(--hb-gray, 100%));
	-webkit-backdrop-filter: blur(var(--hb-blur, 20px)) grayscale(var(--hb-gray, 100%));
	border-radius: 12%;
}
.${REGION_CLASS}.hb-rr-solid {
	backdrop-filter: none;
	-webkit-backdrop-filter: none;
	background: #0f172a;
}
.${REGION_CLASS}.hb-rr-static {
	backdrop-filter: none;
	-webkit-backdrop-filter: none;
	background-color: #0f172a;
	background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url(%23n)' opacity='0.6'/%3E%3C/svg%3E");
	background-size: 140px 140px;
}
.${REGION_CLASS}.hb-rr-pattern {
	backdrop-filter: none;
	-webkit-backdrop-filter: none;
	background-color: #1e293b;
	background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24'%3E%3Cpath d='M12 0L24 12L12 24L0 12Z' fill='none' stroke='%23334155' stroke-width='1.5'/%3E%3C/svg%3E");
	background-size: 24px 24px;
}`;
	portalRoot.appendChild(style);
	(document.documentElement ?? document.body).appendChild(portal);
	bindGlobalListeners();
	return portalRoot;
}

function bindGlobalListeners(): void {
	if (globalListenersBound || typeof document === "undefined") return;
	globalListenersBound = true;
	// Capture phase + passive: catches scrolls on any nested scroll container.
	document.addEventListener("scroll", onViewportChange, {
		capture: true,
		passive: true,
	});
	window.addEventListener("resize", onViewportChange, { passive: true });
}

function onViewportChange(): void {
	scheduleSyncAll();
}

let syncFrameQueued = false;

function scheduleSync(el: HTMLElement): void {
	pendingSync.add(el);
	if (syncFrameQueued) return;
	syncFrameQueued = true;
	const run = () => {
		syncFrameQueued = false;
		flushSync();
	};
	if (typeof requestAnimationFrame === "function") requestAnimationFrame(run);
	else run();
}

function scheduleSyncAll(): void {
	for (const el of entries.keys()) pendingSync.add(el);
	scheduleSync(document.documentElement as unknown as HTMLElement);
}

function flushSync(): void {
	// Prune disconnected elements on every tick (spec: prune + onRemoved).
	for (const el of [...entries.keys()]) {
		if (!el.isConnected) removeRegionBlur(el);
	}
	for (const el of pendingSync) {
		const entry = entries.get(el);
		if (entry != null) syncEntry(entry);
	}
	pendingSync.clear();
}

/** parentElement that crosses open shadow boundaries via the host. */
function parentAcrossShadow(el: Element): Element | null {
	if (el.parentElement != null) return el.parentElement;
	const root = el.getRootNode?.();
	if (root != null && typeof ShadowRoot !== "undefined" && root instanceof ShadowRoot) {
		return root.host.parentElement;
	}
	return null;
}

/** Intersect `rect` with every clipping ancestor's border box. */
function clipRectFor(el: HTMLElement, rect: DOMRect): PixelRect | null {
	let x0 = rect.left;
	let y0 = rect.top;
	let x1 = rect.right;
	let y1 = rect.bottom;
	let node = parentAcrossShadow(el);
	// `document.body` never clips: per CSS Overflow its overflow-* values
	// propagate to the viewport when the root element's are visible, and its
	// border box can be degenerate anyway (YouTube scrolls inside
	// `ytd-page-manager`; body reports h:0). The portal's own fixed inset:0
	// box already provides the viewport clip.
	while (node != null && node !== document.documentElement && node !== document.body) {
		const cs = getComputedStyle(node);
		const clips =
			cs.overflow !== "visible" ||
			cs.overflowX !== "visible" ||
			cs.overflowY !== "visible" ||
			cs.clipPath !== "none";
		if (clips) {
			const r = node.getBoundingClientRect();
			x0 = Math.max(x0, r.left);
			y0 = Math.max(y0, r.top);
			x1 = Math.min(x1, r.right);
			y1 = Math.min(y1, r.bottom);
			if (x1 <= x0 || y1 <= y0) return null;
		}
		node = parentAcrossShadow(node);
	}
	return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/** Painted-pixel content rect (element-box px) for the element's media. */
function contentRectFor(
	el: HTMLElement,
	kind: MediaKind,
	ew: number,
	eh: number,
): PixelRect | null {
	// No computed style outside a DOM (unit tests) — geometry unknowable.
	if (typeof getComputedStyle !== "function") return null;
	const cs = getComputedStyle(el);
	if (kind === "background") {
		const url = extractBackgroundImageUrl(cs.backgroundImage);
		const size = url != null ? NATURAL_SIZE.get(url) : undefined;
		if (url == null || size == null || size.w <= 0 || size.h <= 0) return null;
		return imageContentRect(
			ew,
			eh,
			size.w,
			size.h,
			backgroundFitFor(cs.backgroundSize),
			firstLayer(cs.backgroundPosition),
		);
	}
	let nw = 0;
	let nh = 0;
	if (el instanceof HTMLImageElement) {
		nw = el.naturalWidth;
		nh = el.naturalHeight;
	} else if (el instanceof HTMLVideoElement) {
		nw = el.videoWidth;
		nh = el.videoHeight;
	}
	if (nw <= 0 || nh <= 0) return null;
	return imageContentRect(ew, eh, nw, nh, cs.objectFit, cs.objectPosition);
}

/** (Re)position one entry's host + region children. */
function syncEntry(entry: RegionEntry): void {
	const { el, host } = entry;
	const rect = el.getBoundingClientRect();
	if (rect.width <= 0 || rect.height <= 0) {
		host.style.display = "none";
		return;
	}
	const clip = clipRectFor(el, rect);
	if (clip == null || clip.w <= 0 || clip.h <= 0) {
		host.style.display = "none";
		return;
	}
	// Cheap cull: far offscreen entries skip style writes entirely.
	const vh = window.innerHeight || 0;
	const vw = window.innerWidth || 0;
	if (
		clip.y > vh + OFFSCREEN_MARGIN ||
		clip.y + clip.h < -OFFSCREEN_MARGIN ||
		clip.x > vw + OFFSCREEN_MARGIN ||
		clip.x + clip.w < -OFFSCREEN_MARGIN
	) {
		return;
	}
	const content = contentRectFor(el, entry.kind, rect.width, rect.height);
	if (content == null) {
		host.style.display = "none";
		return;
	}
	host.style.display = "";
	host.style.left = `${clip.x}px`;
	host.style.top = `${clip.y}px`;
	host.style.width = `${clip.w}px`;
	host.style.height = `${clip.h}px`;
	for (let i = 0; i < entry.boxes.length; i++) {
		const region = entry.regions[i];
		const box = entry.boxes[i];
		if (region == null || box == null) continue;
		const r = regionToElementPx(box, content);
		// Element-box px → viewport px → host-relative (host sits at clip).
		region.style.left = `${rect.left + r.x - clip.x}px`;
		region.style.top = `${rect.top + r.y - clip.y}px`;
		region.style.width = `${r.w}px`;
		region.style.height = `${r.h}px`;
	}
}

// ---------------------------------------------------------------------------
// Pending upgrades: media that can't map boxes yet (unloaded img, undecoded
// background URL) gets whole-element blur now and upgrades to region blur
// when its pixels' geometry becomes knowable.
// ---------------------------------------------------------------------------

interface Upgrade {
	el: HTMLElement;
	kind: MediaKind;
	detections: readonly Detection[];
	settings: Settings;
	/** Disconnects the temporary ResizeObserver used for layout upgrades. */
	cleanup?: () => void;
}

const upgrades = new Map<HTMLElement, Upgrade>();
const upgradesByUrl = new Map<string, Set<HTMLElement>>();

function backgroundUrlOf(el: HTMLElement): string | null {
	if (typeof getComputedStyle !== "function") return null;
	return extractBackgroundImageUrl(getComputedStyle(el).backgroundImage);
}

function dropUpgradeIndex(el: HTMLElement): void {
	for (const [url, set] of upgradesByUrl) {
		if (set.delete(el) && set.size === 0) upgradesByUrl.delete(url);
	}
}

function fulfillUpgrade(el: HTMLElement): void {
	const u = upgrades.get(el);
	if (u == null) return;
	upgrades.delete(el);
	dropUpgradeIndex(el);
	u.cleanup?.();
	if (!el.isConnected) return;
	applyBlurMode(el, u.detections, u.kind, activeSettings ?? u.settings);
}

function requestUpgrade(u: Upgrade): void {
	upgrades.set(u.el, u);
	if (u.kind === "background") {
		const url = backgroundUrlOf(u.el);
		if (url != null) {
			let set = upgradesByUrl.get(url);
			if (set == null) {
				set = new Set();
				upgradesByUrl.set(url, set);
			}
			set.add(u.el);
			ensureNaturalSize(url);
			return; // fulfilled by the decode callback
		}
		return; // no URL → can never resolve; stays whole-blurred
	}
	// A zero-size rect (display:none, not yet laid out) resolves neither via
	// load events nor decode — a temporary ResizeObserver catches the element
	// gaining layout.
	if (typeof ResizeObserver === "function") {
		const ro = new ResizeObserver(() => {
			fulfillUpgrade(u.el);
			ro.disconnect();
		});
		ro.observe(u.el);
		u.cleanup = () => ro.disconnect();
	}
	const onReady = () => fulfillUpgrade(u.el);
	if (typeof HTMLImageElement !== "undefined" && u.el instanceof HTMLImageElement) {
		u.el.addEventListener("load", onReady, { once: true });
		u.el.addEventListener("error", onReady, { once: true });
	} else if (typeof HTMLVideoElement !== "undefined" && u.el instanceof HTMLVideoElement) {
		u.el.addEventListener("loadeddata", onReady, { once: true });
		u.el.addEventListener("error", onReady, { once: true });
	}
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Whether `el` currently has a live region overlay. */
export function hasRegionBlur(el: HTMLElement): boolean {
	return entries.has(el);
}

/**
 * Show region blur for `el` covering `boxes` (normalized over image pixels).
 * Returns true when the portal entry is live; false when the element's media
 * geometry isn't knowable yet (caller should whole-blur and may
 * `requestUpgrade` via applyBlurMode's fallback path).
 */
export function applyRegionBlur(
	el: HTMLElement,
	boxes: readonly Detection["box"][],
	kind: MediaKind,
	settings: Settings,
): boolean {
	removeRegionBlur(el);
	if (boxes.length === 0) return false;
	if (typeof el.getBoundingClientRect !== "function") return false;
	const rect = el.getBoundingClientRect();
	if (rect.width <= 0 || rect.height <= 0) return false;
	// Media geometry must be resolvable *now* — otherwise the caller falls
	// back to whole-element blur and we queue an upgrade.
	if (contentRectFor(el, kind, rect.width, rect.height) == null) return false;

	const root = ensurePortal();
	const host = document.createElement("div");
	host.className = HOST_CLASS;
	const styleClass = settings.censorStyle === "blur" ? "" : ` hb-rr-${settings.censorStyle}`;
	const regions = boxes.slice(0, MAX_REGIONS).map(() => {
		const r = document.createElement("div");
		r.className = `${REGION_CLASS}${styleClass}`;
		host.appendChild(r);
		return r;
	});
	root.appendChild(host);

	const entry: RegionEntry = {
		el,
		kind,
		boxes: boxes.slice(0, MAX_REGIONS),
		style: settings.censorStyle,
		host,
		regions,
		resizeObserver: null,
		intersectionObserver: null,
		revealTimer: null,
		hoverReveal: revealAllowed(kind, settings),
	};
	entries.set(el, entry);
	el.dataset.hbRegion = "1";

	if (typeof ResizeObserver === "function") {
		entry.resizeObserver = new ResizeObserver(() => scheduleSync(el));
		entry.resizeObserver.observe(el);
	}
	if (typeof IntersectionObserver === "function") {
		entry.intersectionObserver = new IntersectionObserver((obs) => {
			if (obs[0]?.isIntersecting) scheduleSync(el);
		});
		entry.intersectionObserver.observe(el);
	}
	bindHoverReveal(entry);
	syncEntry(entry);
	// Late reflow (images above finishing, fonts swapping) can shift the
	// element after the verdict; resync once the frame settles.
	requestAnimationFrame(() => {
		if (entries.has(el)) syncEntry(entry);
	});
	return true;
}

function bindHoverReveal(entry: RegionEntry): void {
	const { el, host } = entry;
	const enter = () => {
		if (!entry.hoverReveal) return;
		if (entry.revealTimer != null) clearTimeout(entry.revealTimer);
		entry.revealTimer = setTimeout(() => {
			entry.revealTimer = null;
			host.style.opacity = "0";
		}, REVEAL_DELAY_MS);
	};
	const leave = () => {
		if (entry.revealTimer != null) {
			clearTimeout(entry.revealTimer);
			entry.revealTimer = null;
		}
		host.style.opacity = "";
	};
	el.addEventListener("mouseenter", enter);
	el.addEventListener("mouseleave", leave);
	// Stash cleanups on the entry via a closure map — RegionEntry stays lean.
	entryCleanup.set(el, () => {
		el.removeEventListener("mouseenter", enter);
		el.removeEventListener("mouseleave", leave);
		if (entry.revealTimer != null) clearTimeout(entry.revealTimer);
	});
}

const entryCleanup = new Map<HTMLElement, () => void>();

/** Tear down the portal entry for `el`, if any. Idempotent. */
export function removeRegionBlur(el: HTMLElement): void {
	const entry = entries.get(el);
	if (entry != null) {
		entry.resizeObserver?.disconnect();
		entry.intersectionObserver?.disconnect();
		entryCleanup.get(el)?.();
		entryCleanup.delete(el);
		entry.host.remove();
		entries.delete(el);
	}
	const up = upgrades.get(el);
	if (up != null) {
		up.cleanup?.();
		upgrades.delete(el);
		dropUpgradeIndex(el);
	}
	delete el.dataset.hbRegion;
}

/** Re-sync one entry immediately (e.g. after scroll-into-view). */
export function syncRegionBlur(el: HTMLElement): void {
	const entry = entries.get(el);
	if (entry != null) syncEntry(entry);
}

/** Re-sync every tracked entry. */
export function syncAllRegionBlurs(): void {
	scheduleSyncAll();
}

/** Remove every portal entry (pipeline stop / site toggle). */
export function clearAllRegionBlurs(): void {
	for (const el of [...entries.keys()]) removeRegionBlur(el);
	upgrades.clear();
	upgradesByUrl.clear();
	pendingSync.clear();
}

/** Tear down the portal and all listeners (tests / full teardown). */
export function disposeRegionPortal(): void {
	clearAllRegionBlurs();
	portal?.remove();
	portal = null;
	portalRoot = null;
	globalListenersBound = false;
}

/**
 * Apply the blur mode for a blur verdict: region portal when eligible,
 * whole-element `data-hb-blur` otherwise. Full-frame detections (NSFW
 * suggestive/explicit, or any box covering ~the whole frame) always take the
 * whole-element path — there is no sub-box to localize.
 */
export function applyBlurMode(
	el: HTMLElement,
	detections: readonly Detection[],
	kind: MediaKind,
	settings: Settings,
): void {
	removeRegionBlur(el);
	delete el.dataset.hbRegion;
	setBlurred(el, false);
	const plan = planCensor(detections, kind, settings);
	// Non-blur styles can't use the element's CSS filter (it only blurs) —
	// a whole-element censor goes through the portal as one full-frame box.
	if (plan.whole && settings.censorStyle !== "blur") {
		if (applyRegionBlur(el, [FULL_FRAME_BOX], kind, settings)) return;
		requestUpgrade({ el, kind, detections, settings });
		setBlurred(el, true); // geometry never resolves → blur fallback
		return;
	}
	if (!plan.whole && applyRegionBlur(el, plan.boxes, kind, settings)) {
		return;
	}
	// Region apply failed — likely unloaded media or undecoded bg URL.
	// Whole-blur now; queue an upgrade for when geometry resolves. Only
	// when region mode was actually intended (plan.whole skips this).
	if (!plan.whole && blurCapableDetections(detections, settings).length > 0) {
		requestUpgrade({ el, kind, detections, settings });
	}
	setBlurred(el, true);
}
