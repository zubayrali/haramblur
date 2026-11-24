// @vitest-environment jsdom
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import {
	applyBlurMode,
	backgroundFitFor,
	blurRegionBoxes,
	disposeRegionPortal,
	hasRegionBlur,
	imageContentRect,
	isFullFrameBox,
	naturalSizeFor,
	padBox,
	parseObjectPosition,
	regionToElementPx,
	removeRegionBlur,
	syncAllRegionBlurs,
} from "../../src/content/region";
import type { Detection } from "../../src/shared/protocol";
import { DEFAULT_SETTINGS, type Settings } from "../../src/shared/settings";

const SETTINGS: Settings = { ...DEFAULT_SETTINGS, regionBlur: true };

function det(label: Detection["label"], score: number, box: Detection["box"]): Detection {
	return { label, score, box };
}

const FACE_BOX: Detection["box"] = [0.1, 0.1, 0.2, 0.2];

// jsdom's getComputedStyle doesn't resolve shorthand defaults the way
// browsers do — patch only the *empty* values region.ts reads so unset
// overflow/clip-path don't read as clipping ancestors.
const realGCS = globalThis.getComputedStyle;
beforeAll(() => {
	vi.stubGlobal("getComputedStyle", (el: Element, pseudoElt?: string | null) => {
		const cs = realGCS(el, pseudoElt);
		const fill = (prop: string, cssName: string, value: string) => {
			if (cs.getPropertyValue(cssName) !== "") return;
			try {
				Object.defineProperty(cs, prop, { value, configurable: true });
			} catch {
				// read-only declaration — leave as-is
			}
		};
		fill("overflow", "overflow", "visible");
		fill("overflowX", "overflow-x", "visible");
		fill("overflowY", "overflow-y", "visible");
		fill("clipPath", "clip-path", "none");
		return cs;
	});
});
afterAll(() => vi.unstubAllGlobals());
afterEach(() => {
	disposeRegionPortal();
	document.body.innerHTML = "";
});

function stubRect(
	el: Element,
	r: { left: number; top: number; width: number; height: number },
): void {
	el.getBoundingClientRect = () =>
		({
			left: r.left,
			top: r.top,
			right: r.left + r.width,
			bottom: r.top + r.height,
			width: r.width,
			height: r.height,
			x: r.left,
			y: r.top,
			toJSON: () => ({}),
		}) as DOMRect;
}

function portalShadow(): ShadowRoot {
	const portal = document.getElementById("hb-region-portal");
	expect(portal).not.toBeNull();
	const root = portal?.shadowRoot;
	expect(root).not.toBeNull();
	return root as ShadowRoot;
}

function regionHosts(): HTMLElement[] {
	return [...portalShadow().querySelectorAll<HTMLElement>(".hb-rh")];
}

describe("pure geometry", () => {
	it("padBox grows by REGION_PAD and clamps to 0..1", () => {
		expect(padBox([0.5, 0.5, 0.2, 0.2])).toEqual([0.46, 0.46, 0.28, 0.28]);
		expect(padBox([0, 0, 0.05, 0.05])).toEqual([0, 0, 0.09, 0.09]);
		expect(padBox([0.9, 0.9, 0.2, 0.2])).toEqual([0.86, 0.86, 0.14, 0.14]);
	});

	it("isFullFrameBox detects whole-frame detections", () => {
		expect(isFullFrameBox([0, 0, 1, 1])).toBe(true);
		expect(isFullFrameBox([0.01, 0.01, 0.98, 0.98])).toBe(true);
		expect(isFullFrameBox([0.1, 0.1, 0.5, 0.5])).toBe(false);
	});

	it("parseObjectPosition handles keywords, percentages, single values", () => {
		expect(parseObjectPosition("left top")).toEqual([0, 0]);
		expect(parseObjectPosition("right bottom")).toEqual([1, 1]);
		expect(parseObjectPosition("center")).toEqual([0.5, 0.5]);
		expect(parseObjectPosition("25% 75%")).toEqual([0.25, 0.75]);
		expect(parseObjectPosition("top")).toEqual([0.5, 0]);
		expect(parseObjectPosition("")).toEqual([0.5, 0.5]);
	});

	it("imageContentRect: fill covers, contain letterboxes, cover crops", () => {
		// 200x100 element, 400x200 image → same aspect: identical either way.
		expect(imageContentRect(200, 100, 400, 200, "fill", "")).toEqual({
			x: 0,
			y: 0,
			w: 200,
			h: 100,
		});
		// contain: 400x400 into 200x100 → 100x100 centered vertically.
		expect(imageContentRect(200, 100, 400, 400, "contain", "center")).toEqual({
			x: 50,
			y: 0,
			w: 100,
			h: 100,
		});
		// cover: same inputs → 200x200 cropped, centered → y=-50.
		expect(imageContentRect(200, 100, 400, 400, "cover", "center")).toEqual({
			x: 0,
			y: -50,
			w: 200,
			h: 200,
		});
		// none: natural size at position anchor.
		expect(imageContentRect(200, 100, 400, 200, "none", "left top")).toEqual({
			x: 0,
			y: 0,
			w: 400,
			h: 200,
		});
	});

	it("regionToElementPx maps normalized image boxes into element pixels", () => {
		const content = { x: 50, y: 0, w: 100, h: 100 };
		expect(regionToElementPx([0.1, 0.2, 0.3, 0.4], content)).toEqual({
			x: 60,
			y: 20,
			w: 30,
			h: 40,
		});
	});

	it("backgroundFitFor maps CSS background-size to fit vocabulary", () => {
		expect(backgroundFitFor("cover")).toBe("cover");
		expect(backgroundFitFor("contain")).toBe("contain");
		expect(backgroundFitFor("auto")).toBe("none");
		expect(backgroundFitFor("auto 100px")).toBe("none");
		expect(backgroundFitFor("100% 100%")).toBe("fill");
		expect(backgroundFitFor("cover, contain")).toBe("cover");
	});

	it("blurRegionBoxes keeps only blur-capable detections, sorted and padded", () => {
		const detections = [
			det("FEMALE_BREAST_EXPOSED", 0.7, [0.5, 0.5, 0.1, 0.1]),
			det("FEMALE_BREAST_EXPOSED", 0.9, FACE_BOX),
			// Not blur-capable under defaults: male face with blurMale off.
			det("FACE_MALE", 0.99, [0.7, 0.7, 0.1, 0.1]),
		];
		const boxes = blurRegionBoxes(detections, SETTINGS);
		expect(boxes).toHaveLength(2);
		// Highest score first → FACE_BOX (0.9) padded.
		expect(boxes[0]).toEqual(padBox(FACE_BOX));
		expect(boxes[1]).toEqual(padBox([0.5, 0.5, 0.1, 0.1]));
	});
});

describe("portal lifecycle", () => {
	function makeImg(): HTMLImageElement {
		const img = document.createElement("img");
		img.src = "https://img.test/photo.jpg";
		Object.defineProperty(img, "naturalWidth", { value: 400, configurable: true });
		Object.defineProperty(img, "naturalHeight", { value: 200, configurable: true });
		stubRect(img, { left: 100, top: 50, width: 200, height: 100 });
		document.body.appendChild(img);
		return img;
	}

	it("creates one portal host with a region child per box", () => {
		const img = makeImg();
		const detections = [
			det("FEMALE_BREAST_EXPOSED", 0.9, FACE_BOX),
			det("FEMALE_GENITALIA_EXPOSED", 0.8, [0.6, 0.6, 0.1, 0.1]),
		];
		applyBlurMode(img, detections, "image", SETTINGS);

		expect(hasRegionBlur(img)).toBe(true);
		expect(img.dataset.hbRegion).toBe("1");
		// Region mode must NOT set the whole-element blur attribute.
		expect(img.dataset.hbBlur).toBeUndefined();

		const hosts = regionHosts();
		expect(hosts).toHaveLength(1);
		const host = hosts[0] as HTMLElement;
		expect(host.style.left).toBe("100px");
		expect(host.style.top).toBe("50px");
		expect(host.style.width).toBe("200px");
		expect(host.style.height).toBe("100px");

		// Default fit (fill) → content == element box; box [0.1,0.1,0.2,0.2]
		// padded to [0.06,0.06,0.28,0.28] → 12px,6px,56px,28px in element px.
		const regions = host.querySelectorAll<HTMLElement>(".hb-rr");
		expect(regions).toHaveLength(2);
		expect(regions[0]?.style.left).toBe("12px");
		expect(regions[0]?.style.top).toBe("6px");
		expect(regions[0]?.style.width).toBe("56px");
		expect(regions[0]?.style.height).toBe("28px");
	});

	it("whole-element blur for full-frame detections (no portal entry)", () => {
		const img = makeImg();
		applyBlurMode(img, [det("NSFW_EXPLICIT", 0.9, [0, 0, 1, 1])], "image", SETTINGS);
		expect(img.dataset.hbBlur).toBe("1");
		expect(hasRegionBlur(img)).toBe(false);
		expect(document.getElementById("hb-region-portal")).toBeNull();
	});

	it("whole-element blur when regionBlur is off", () => {
		const img = makeImg();
		applyBlurMode(img, [det("FEMALE_BREAST_EXPOSED", 0.9, FACE_BOX)], "image", {
			...SETTINGS,
			regionBlur: false,
		});
		expect(img.dataset.hbBlur).toBe("1");
		expect(hasRegionBlur(img)).toBe(false);
	});

	it("caps region children at MAX_REGIONS", () => {
		const img = makeImg();
		const detections = Array.from({ length: 20 }, (_, i) =>
			det("FEMALE_BREAST_EXPOSED", 0.9 - i * 0.01, [0.05 + i * 0.04, 0.05, 0.05, 0.05]),
		);
		applyBlurMode(img, detections, "image", SETTINGS);
		const regions = regionHosts()[0]?.querySelectorAll(".hb-rr");
		expect(regions).toHaveLength(16);
	});

	it("clips the host to an overflow:hidden ancestor", () => {
		const clipper = document.createElement("div");
		clipper.style.overflow = "hidden";
		stubRect(clipper, { left: 120, top: 60, width: 100, height: 80 });
		document.body.appendChild(clipper);
		const img = document.createElement("img");
		img.src = "https://img.test/clip.jpg";
		Object.defineProperty(img, "naturalWidth", { value: 400, configurable: true });
		Object.defineProperty(img, "naturalHeight", { value: 200, configurable: true });
		// Image overflows the clipper on every side.
		stubRect(img, { left: 100, top: 50, width: 200, height: 100 });
		clipper.appendChild(img);

		applyBlurMode(img, [det("FEMALE_BREAST_EXPOSED", 0.9, FACE_BOX)], "image", SETTINGS);
		const host = regionHosts()[0] as HTMLElement;
		// clip = img ∩ clipper = {120,60,100,80}
		expect(host.style.left).toBe("120px");
		expect(host.style.top).toBe("60px");
		expect(host.style.width).toBe("100px");
		expect(host.style.height).toBe("80px");
	});

	it("prunes disconnected elements on the next sync", async () => {
		const img = makeImg();
		applyBlurMode(img, [det("FEMALE_BREAST_EXPOSED", 0.9, FACE_BOX)], "image", SETTINGS);
		expect(hasRegionBlur(img)).toBe(true);
		img.remove();
		syncAllRegionBlurs();
		// The flush is rAF-throttled when rAF exists — poll either way.
		await vi.waitFor(() => expect(hasRegionBlur(img)).toBe(false));
		expect(regionHosts()).toHaveLength(0);
	});

	it("removeRegionBlur tears down host and attributes", () => {
		const img = makeImg();
		applyBlurMode(img, [det("FEMALE_BREAST_EXPOSED", 0.9, FACE_BOX)], "image", SETTINGS);
		removeRegionBlur(img);
		expect(hasRegionBlur(img)).toBe(false);
		expect(img.dataset.hbRegion).toBeUndefined();
		expect(regionHosts()).toHaveLength(0);
		// Idempotent.
		removeRegionBlur(img);
	});

	it("hover reveal fades the host only when the kind allows it", async () => {
		vi.useFakeTimers();
		try {
			const img = makeImg();
			// unblurImages defaults to false — reveal is opt-in.
			applyBlurMode(img, [det("FEMALE_BREAST_EXPOSED", 0.9, FACE_BOX)], "image", {
				...SETTINGS,
				unblurImages: true,
			});
			const host = regionHosts()[0] as HTMLElement;
			img.dispatchEvent(new Event("mouseenter"));
			vi.advanceTimersByTime(1_100);
			expect(host.style.opacity).toBe("0");
			img.dispatchEvent(new Event("mouseleave"));
			expect(host.style.opacity).toBe("");
		} finally {
			vi.useRealTimers();
		}
	});

	it("hover reveal stays hidden when unblurImages is off", async () => {
		vi.useFakeTimers();
		try {
			const img = makeImg();
			applyBlurMode(img, [det("FEMALE_BREAST_EXPOSED", 0.9, FACE_BOX)], "image", {
				...SETTINGS,
				unblurImages: false,
			});
			const host = regionHosts()[0] as HTMLElement;
			img.dispatchEvent(new Event("mouseenter"));
			vi.advanceTimersByTime(1_100);
			expect(host.style.opacity).toBe("");
		} finally {
			vi.useRealTimers();
		}
	});
});

describe("background-image upgrade path", () => {
	it("whole-blurs until the bg image decodes, then upgrades to region blur", async () => {
		// Stub Image so ensureNaturalSize's decode resolves immediately.
		const RealImage = globalThis.Image;
		class FakeImage {
			naturalWidth = 800;
			naturalHeight = 600;
			onload: (() => void) | null = null;
			onerror: (() => void) | null = null;
			set src(_v: string) {
				queueMicrotask(() => this.onload?.());
			}
		}
		vi.stubGlobal("Image", FakeImage);
		try {
			const div = document.createElement("div");
			div.style.backgroundImage = 'url("https://bg.test/hero.png")';
			stubRect(div, { left: 10, top: 20, width: 300, height: 200 });
			document.body.appendChild(div);

			applyBlurMode(div, [det("FEMALE_BREAST_EXPOSED", 0.9, FACE_BOX)], "background", SETTINGS);
			// Natural size unknown → whole-element blur + pending upgrade.
			expect(div.dataset.hbBlur).toBe("1");
			expect(hasRegionBlur(div)).toBe(false);

			await vi.waitFor(() => expect(hasRegionBlur(div)).toBe(true));
			expect(div.dataset.hbBlur).toBeUndefined();
			expect(naturalSizeFor("https://bg.test/hero.png")).toEqual({ w: 800, h: 600 });
			const host = regionHosts()[0] as HTMLElement;
			expect(host.style.left).toBe("10px");
			expect(host.style.top).toBe("20px");
		} finally {
			vi.stubGlobal("Image", RealImage);
		}
	});
});
