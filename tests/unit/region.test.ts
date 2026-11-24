import { describe, expect, it } from "vitest";
import {
	blurRegionBoxes,
	imageContentRect,
	padBox,
	parseObjectPosition,
	regionToElementPx,
} from "../../src/content/region";
import type { Detection } from "../../src/shared/protocol";
import { DEFAULT_SETTINGS } from "../../src/shared/settings";

const det = (
	label: Detection["label"],
	score: number,
	box: Detection["box"] = [0.1, 0.1, 0.2, 0.2],
): Detection => ({
	label,
	score,
	box,
});

describe("padBox", () => {
	it("grows the box on every side", () => {
		expect(padBox([0.2, 0.3, 0.4, 0.5], 0.1)).toEqual([0.1, 0.2, 0.6, 0.7]);
	});

	it("clamps to the unit square", () => {
		expect(padBox([0, 0, 0.05, 0.05], 0.1)).toEqual([0, 0, 0.15, 0.15]);
		expect(padBox([0.9, 0.9, 0.1, 0.1], 0.1)).toEqual([0.8, 0.8, 0.2, 0.2]);
	});
});

describe("blurRegionBoxes", () => {
	it("keeps only detections that meet the blur bar", () => {
		const boxes = blurRegionBoxes(
			[
				det("FACE_FEMALE", 0.9, [0.1, 0.1, 0.2, 0.2]),
				det("BELLY_COVERED", 0.99, [0.5, 0.5, 0.2, 0.2]), // covered: never blurs
				det("FACE_MALE", 0.99, [0.7, 0.7, 0.1, 0.1]), // blurMale off by default
			],
			DEFAULT_SETTINGS,
		);
		expect(boxes).toHaveLength(1);
		// Padded by REGION_PAD (0.04).
		expect(boxes[0]).toEqual([0.06, 0.06, 0.28, 0.28]);
	});

	it("respects the gender toggles like decide()", () => {
		const detections = [det("FACE_FEMALE", 0.9), det("FACE_MALE", 0.9)];
		// blurMale defaults off, so each single-toggle case enables the other.
		expect(
			blurRegionBoxes(detections, {
				...DEFAULT_SETTINGS,
				blurFemale: false,
				blurMale: true,
			}),
		).toHaveLength(1);
		expect(
			blurRegionBoxes(detections, {
				...DEFAULT_SETTINGS,
				blurFemale: true,
				blurMale: true,
			}),
		).toHaveLength(2);
		expect(
			blurRegionBoxes(detections, {
				...DEFAULT_SETTINGS,
				blurFemale: false,
				blurMale: false,
			}),
		).toHaveLength(0);
	});

	it("returns nothing when the extension is off", () => {
		expect(
			blurRegionBoxes([det("ANUS_EXPOSED", 0.99)], { ...DEFAULT_SETTINGS, status: false }),
		).toEqual([]);
	});

	it("sorts by score descending and caps the region count", () => {
		const many = Array.from({ length: 20 }, (_, i) =>
			det("FEMALE_BREAST_EXPOSED", 0.5 + i / 100, [i / 25, 0, 0.01, 0.01]),
		);
		const boxes = blurRegionBoxes(many, DEFAULT_SETTINGS);
		expect(boxes).toHaveLength(16);
		// Highest score first → i=19, x = 19/25, padded by REGION_PAD (0.04).
		expect(boxes[0]?.[0]).toBeCloseTo(19 / 25 - 0.04, 5);
	});
});

describe("parseObjectPosition", () => {
	it("parses keywords", () => {
		expect(parseObjectPosition("left top")).toEqual([0, 0]);
		expect(parseObjectPosition("right bottom")).toEqual([1, 1]);
		expect(parseObjectPosition("center")).toEqual([0.5, 0.5]);
		expect(parseObjectPosition("left")).toEqual([0, 0.5]);
	});

	it("parses percentages", () => {
		expect(parseObjectPosition("25% 75%")).toEqual([0.25, 0.75]);
		expect(parseObjectPosition("0% 100%")).toEqual([0, 1]);
		// A lone percentage centers the other axis.
		expect(parseObjectPosition("25%")).toEqual([0.25, 0.5]);
	});

	it("a lone vertical keyword still sets y", () => {
		expect(parseObjectPosition("top")).toEqual([0.5, 0]);
		expect(parseObjectPosition("bottom")).toEqual([0.5, 1]);
	});

	it("degrades unknown values to center", () => {
		expect(parseObjectPosition("")).toEqual([0.5, 0.5]);
		expect(parseObjectPosition("10px 20px")).toEqual([0.5, 0.5]);
	});
});

describe("imageContentRect", () => {
	// Element 200×100, natural image 100×100.
	it("fill covers the whole element box", () => {
		expect(imageContentRect(200, 100, 100, 100, "fill", "center")).toEqual({
			x: 0,
			y: 0,
			h: 100,
			w: 200,
		});
	});

	it("cover crops to fill — content overflows equally on the long axis", () => {
		// scale = max(200/100, 100/100) = 2 → content 200×200, y centered → -50.
		expect(imageContentRect(200, 100, 100, 100, "cover", "center")).toEqual({
			x: 0,
			y: -50,
			h: 200,
			w: 200,
		});
	});

	it("cover honors object-position", () => {
		// "top" → y=0 → crop only at the bottom.
		expect(imageContentRect(200, 100, 100, 100, "cover", "center top")).toEqual({
			x: 0,
			y: 0,
			h: 200,
			w: 200,
		});
	});

	it("contain letterboxes — content centered inside the element", () => {
		// scale = min(2, 1) = 1 → content 100×100 centered in 200×100 → x=50.
		expect(imageContentRect(200, 100, 100, 100, "contain", "center")).toEqual({
			x: 50,
			y: 0,
			h: 100,
			w: 100,
		});
	});

	it("none renders at natural size, anchored by position", () => {
		expect(imageContentRect(200, 100, 50, 50, "none", "left top")).toEqual({
			x: 0,
			y: 0,
			h: 50,
			w: 50,
		});
	});

	it("scale-down never upscales", () => {
		// Element smaller than natural → same as contain; larger → natural size.
		expect(imageContentRect(50, 50, 100, 100, "scale-down", "center")).toEqual({
			x: 0,
			y: 0,
			h: 50,
			w: 50,
		});
		expect(imageContentRect(200, 200, 100, 100, "scale-down", "center")).toEqual({
			x: 50,
			y: 50,
			h: 100,
			w: 100,
		});
	});
});

describe("regionToElementPx", () => {
	it("maps a normalized box through the content rect", () => {
		const content = { x: 10, y: 20, w: 100, h: 50 };
		expect(regionToElementPx([0.5, 0.5, 0.2, 0.4], content)).toEqual({
			x: 60,
			y: 45,
			w: 20,
			h: 20,
		});
	});
});
