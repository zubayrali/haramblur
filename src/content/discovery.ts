/**
 * Media discovery: a MutationObserver (childList + the attributes that carry
 * media URLs), an IntersectionObserver that marks visibility, and an initial
 * DOM scan (docs/PLAN.md WP2, "Blur what the user sees").
 *
 * Emits {@link MediaCandidate}s; the orchestrator decides what to blur and
 * classify. `<iframe>` content needs no code here — the content script runs
 * in every frame (`allFrames: true`).
 */

/** Attributes that can introduce or change a media URL. */
const OBSERVED_ATTRIBUTES = ["src", "srcset", "poster", "style", "href", "data-src"];

/** `data:` URLs shorter than this are icons or tracking payloads — skip. */
export const DATA_URL_MIN_LENGTH = 1024;

/** Media smaller than this on both axes is a tracking pixel — skip. */
export const TRACKING_PIXEL_MAX_PX = 24;

/** Tag names that realistically carry a CSS background image. */
const BG_TAG_SELECTOR = [
	"div",
	"section",
	"article",
	"main",
	"header",
	"footer",
	"aside",
	"nav",
	"figure",
	"span",
	"a",
	"li",
	"td",
	"th",
].join(",");

const CANDIDATE_SELECTOR = `img, video, ${BG_TAG_SELECTOR}, [style*="url("]`;

export type MediaKind = "image" | "video" | "background";

export interface MediaCandidate {
	/** The element the blur applies to. */
	element: HTMLElement;
	kind: MediaKind;
	/** Resolved URL to classify; `null` until the element has one. */
	url: string | null;
	width: number;
	height: number;
	/** Latest known visibility (IntersectionObserver keeps it updated). */
	visible: boolean;
}

export interface DiscoveryHandlers {
	/** A new or updated media element was found. */
	onCandidate(candidate: MediaCandidate): void;
	/** An already-found element became visible — boost its queue priority. */
	onVisible(element: HTMLElement): void;
}

export interface DiscoveryHandle {
	stop(): void;
}

/**
 * Whether `url` is worth classifying: non-empty, not a tiny `data:` payload,
 * not SVG (vector content can't carry photo nudity).
 */
export function isClassifiableUrl(url: string | null | undefined): url is string {
	if (url == null || url.length === 0) return false;
	if (url.startsWith("data:")) {
		if (url.length < DATA_URL_MIN_LENGTH) return false;
		const mimeEnd = url.indexOf(",");
		const mime = url.slice(5, mimeEnd > 0 ? mimeEnd : undefined).toLowerCase();
		if (mime.startsWith("image/svg")) return false;
		return mime.startsWith("image/") || mime.startsWith("video/");
	}
	try {
		const pathname = new URL(url, document.baseURI).pathname.toLowerCase();
		if (pathname.endsWith(".svg")) return false;
	} catch {
		return false;
	}
	return true;
}

/** Rendered size on both axes under the tracking-pixel bound. */
export function isTrackingPixel(width: number, height: number): boolean {
	return width > 0 && height > 0 && width < TRACKING_PIXEL_MAX_PX && height < TRACKING_PIXEL_MAX_PX;
}

/** Extract the first `url(...)` from a computed or inline `background-image`. */
export function extractBackgroundImageUrl(cssText: string): string | null {
	const match = /url\(\s*(['"]?)(.*?)\1\s*\)/i.exec(cssText);
	if (match == null || match[2] === undefined || match[2].length === 0) return null;
	try {
		return new URL(match[2], document.baseURI).href;
	} catch {
		return null;
	}
}

/**
 * The URL an `<img>` is actually showing: `currentSrc` (after srcset/picture
 * resolution), then `src`, then a lazy-loader `data-src`.
 */
export function imageUrl(el: HTMLImageElement): string | null {
	return el.currentSrc || el.src || el.dataset.src || null;
}

/** The URL a `<video>` can be classified by: `poster`, then `currentSrc`/`src`. */
export function videoUrl(el: HTMLVideoElement): string | null {
	return el.poster || el.currentSrc || el.src || el.dataset.src || null;
}

function elementSize(el: HTMLElement): { width: number; height: number } {
	const rect = el.getBoundingClientRect();
	return { width: rect.width, height: rect.height };
}

function imgSize(el: HTMLImageElement): { width: number; height: number } {
	// Rendered size is what the user sees; a 1x1 natural image stretched to
	// 400x200 is not a tracking pixel. 0x0 (still loading) is kept.
	const rect = el.getBoundingClientRect();
	const width = rect.width || el.width || el.naturalWidth;
	const height = rect.height || el.height || el.naturalHeight;
	return { width, height };
}

/**
 * Build a {@link MediaCandidate} from an element, or `null` when it should be
 * ignored (tracking pixel, tiny `data:` URL, SVG, no background URL).
 */
export function candidateFor(el: HTMLElement, visible: boolean): MediaCandidate | null {
	let kind: MediaKind;
	let url: string | null;
	let width: number;
	let height: number;
	if (el instanceof HTMLImageElement) {
		kind = "image";
		url = imageUrl(el);
		({ width, height } = imgSize(el));
	} else if (el instanceof HTMLVideoElement) {
		kind = "video";
		url = videoUrl(el);
		({ width, height } = elementSize(el));
	} else {
		const bg = getComputedStyle(el).backgroundImage;
		if (bg === "none" || bg.length === 0) return null;
		url = extractBackgroundImageUrl(bg);
		if (url == null) return null;
		kind = "background";
		({ width, height } = elementSize(el));
	}
	if (isTrackingPixel(width, height)) return null;
	if (url != null && !isClassifiableUrl(url)) return null;
	return { element: el, kind, url, width, height, visible };
}

/** Elements that may produce a candidate: media elements + possible bg carriers. */
export function* mediaElements(root: ParentNode): Generator<HTMLElement> {
	const seen = new Set<Element>();
	function* push(el: Element): Generator<HTMLElement> {
		if (seen.has(el)) return;
		seen.add(el);
		yield el as HTMLElement;
	}
	if (root instanceof Element && root.matches(CANDIDATE_SELECTOR)) yield* push(root);
	for (const el of root.querySelectorAll("img, video")) yield* push(el);
	for (const el of root.querySelectorAll(`${BG_TAG_SELECTOR}, [style*="url("]`)) {
		yield* push(el);
	}
}

export function startDiscovery(root: Document, handlers: DiscoveryHandlers): DiscoveryHandle {
	const visible = new WeakSet<Element>();
	const io = new IntersectionObserver(
		(entries) => {
			for (const entry of entries) {
				const el = entry.target as HTMLElement;
				if (entry.isIntersecting) {
					visible.add(el);
					el.dataset.hbVisible = "1";
					handlers.onVisible(el);
				} else {
					visible.delete(el);
					delete el.dataset.hbVisible;
				}
			}
		},
		{ threshold: 0 },
	);

	function emit(el: HTMLElement): void {
		const candidate = candidateFor(el, visible.has(el));
		if (candidate == null) return;
		io.observe(el);
		handlers.onCandidate(candidate);
	}

	function scan(node: Node): void {
		if (node instanceof Element || node instanceof Document) {
			for (const el of mediaElements(node)) emit(el);
		}
	}

	const mo = new MutationObserver((mutations) => {
		for (const mutation of mutations) {
			if (mutation.type === "childList") {
				for (const node of mutation.addedNodes) {
					if (node instanceof Element) scan(node);
				}
			} else if (mutation.type === "attributes") {
				const target = mutation.target;
				if (!(target instanceof Element)) continue;
				// A <source> change re-resolves the parent <picture>/<video>'s URL.
				if (target instanceof HTMLSourceElement) {
					const parent = target.parentElement;
					if (parent instanceof HTMLPictureElement) {
						const img = parent.querySelector("img");
						if (img != null) emit(img);
					} else if (parent instanceof HTMLVideoElement) {
						emit(parent);
					}
					continue;
				}
				if (target instanceof HTMLElement) emit(target);
			}
		}
	});

	mo.observe(root.documentElement, {
		childList: true,
		subtree: true,
		attributes: true,
		attributeFilter: OBSERVED_ATTRIBUTES,
	});

	// Initial scan once a body exists; document_start scripts may run before it.
	const scanAll = () => scan(root.documentElement);
	if (root.body != null) {
		scanAll();
	} else {
		const early = new MutationObserver((_mutations, observer) => {
			if (root.body != null) {
				observer.disconnect();
				scanAll();
			}
		});
		early.observe(root.documentElement, { childList: true, subtree: true });
	}

	return {
		stop() {
			mo.disconnect();
			io.disconnect();
		},
	};
}
