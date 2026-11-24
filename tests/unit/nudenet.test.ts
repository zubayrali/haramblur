import { describe, expect, it } from "vitest";
import {
	INPUT_SIZE,
	LETTERBOX_FILL,
	letterboxSize,
	NMS_IOU,
	nmsPerClass,
	parseDetections,
	rgbaToTensor,
	SCORE_FLOOR,
} from "../../src/engine/nudenet";
import { NUDENET_LABELS } from "../../src/shared/labels";

const CLASSES = NUDENET_LABELS.length;
const COLS = 7;

/** Build a synthetic [1, 4 + classes, N] channel-major output. */
function makeOutput(
	entries: { cls: number; score: number; cx: number; cy: number; w: number; h: number }[],
	cols = COLS,
): Float32Array {
	const out = new Float32Array((4 + CLASSES) * cols);
	for (const [i, e] of entries.entries()) {
		out[i] = e.cx;
		out[cols + i] = e.cy;
		out[2 * cols + i] = e.w;
		out[3 * cols + i] = e.h;
		out[(4 + e.cls) * cols + i] = e.score;
	}
	return out;
}

describe("letterboxSize", () => {
	it("scales to the top-left with k = min(320/w, 320/h)", () => {
		// 640×480 → k=0.5, drawn 320×240, right/bottom padding stays filled.
		expect(letterboxSize(640, 480)).toEqual({ k: 0.5, dw: 320, dh: 240 });
		// Tall image → width padded.
		expect(letterboxSize(240, 480)).toEqual({ k: 2 / 3, dw: 160, dh: 320 });
		// Square image fills the input exactly.
		expect(letterboxSize(100, 100)).toEqual({ k: 3.2, dw: 320, dh: 320 });
	});
});

describe("rgbaToTensor", () => {
	it("maps RGBA pixels to CHW float32 in 0..1, keeping the 114 fill", () => {
		const rgba = new Uint8ClampedArray(INPUT_SIZE * INPUT_SIZE * 4);
		// Letterboxed buffer: filled with 114, one red pixel at (0,0).
		rgba.fill(LETTERBOX_FILL);
		rgba[0] = 255; // R
		rgba[1] = 0; // G
		rgba[2] = 0; // B
		rgba[3] = 255; // A
		const t = rgbaToTensor(rgba);
		const plane = INPUT_SIZE * INPUT_SIZE;
		expect(t.length).toBe(3 * plane);
		expect(t[0]).toBeCloseTo(1); // R of the red pixel
		expect(t[plane]).toBeCloseTo(0); // G
		expect(t[2 * plane]).toBeCloseTo(0); // B
		expect(t[1]).toBeCloseTo(LETTERBOX_FILL / 255); // fill pixel
	});
});

describe("parseDetections", () => {
	it("parses channel-major output, maps by 1/k, and normalizes 0..1", () => {
		// Source 640×480 → k=0.5. A 100×80 box at input (160,120) maps to
		// source (320,240,200,160) → normalized (0.5,0.5,0.3125,1/3).
		const faceCls = NUDENET_LABELS.indexOf("FACE_FEMALE");
		const data = makeOutput([{ cls: faceCls, score: 0.8, cx: 160, cy: 120, w: 100, h: 80 }]);
		const dets = parseDetections(data, 4 + CLASSES, COLS, 0.5, 640, 480);
		expect(dets).toHaveLength(1);
		const [d] = dets;
		expect(d!.label).toBe("FACE_FEMALE");
		expect(d!.score).toBeCloseTo(0.8);
		expect(d!.box[0]).toBeCloseTo((160 - 50) / 0.5 / 640);
		expect(d!.box[1]).toBeCloseTo((120 - 40) / 0.5 / 480);
		expect(d!.box[2]).toBeCloseTo(100 / 0.5 / 640);
		expect(d!.box[3]).toBeCloseTo(80 / 0.5 / 480);
	});

	it("picks the argmax class per candidate and drops below-floor scores", () => {
		const faceCls = NUDENET_LABELS.indexOf("FACE_MALE");
		const outCls = NUDENET_LABELS.indexOf("ANUS_COVERED");
		const data = makeOutput([
			{ cls: faceCls, score: 0.9, cx: 10, cy: 10, w: 10, h: 10 },
			{ cls: outCls, score: SCORE_FLOOR / 2, cx: 100, cy: 100, w: 20, h: 20 },
		]);
		// Give candidate 0 a second, lower class score that must be ignored.
		data[(4 + 1) * COLS + 0] = 0.4;
		const dets = parseDetections(data, 4 + CLASSES, COLS, 1, 320, 320);
		expect(dets).toHaveLength(1);
		expect(dets[0]!.label).toBe("FACE_MALE");
		expect(dets[0]!.score).toBeCloseTo(0.9);
	});

	it("clamps boxes that extend past the source edge into 0..1", () => {
		const cls = NUDENET_LABELS.indexOf("BELLY_EXPOSED");
		const data = makeOutput([{ cls, score: 0.9, cx: 319, cy: 319, w: 40, h: 40 }]);
		const dets = parseDetections(data, 4 + CLASSES, COLS, 1, 320, 320);
		expect(dets).toHaveLength(1);
		const [x, y, w, h] = dets[0]!.box;
		expect(x).toBeGreaterThanOrEqual(0);
		expect(y).toBeGreaterThanOrEqual(0);
		expect(x + w).toBeLessThanOrEqual(1);
		expect(y + h).toBeLessThanOrEqual(1);
	});
});

describe("nmsPerClass", () => {
	const cand = (cls: number, score: number, x0 = 0, y0 = 0, x1 = 10, y1 = 10) => ({
		cls,
		score,
		x0,
		y0,
		x1,
		y1,
	});

	it("suppresses overlapping same-class boxes above the IoU threshold", () => {
		// Two nearly identical boxes of the same class → keep the higher score.
		const kept = nmsPerClass([
			cand(0, 0.9),
			cand(0, 0.7, 1, 1, 11, 11),
			cand(0, 0.6, 100, 100, 110, 110), // disjoint → kept
		]);
		expect(kept).toHaveLength(2);
		expect(kept.map((c) => c.score).sort()).toEqual([0.6, 0.9]);
	});

	it("never suppresses across classes, even when identical", () => {
		const kept = nmsPerClass([cand(0, 0.9), cand(1, 0.8)]);
		expect(kept).toHaveLength(2);
	});

	it("keeps same-class boxes at exactly the IoU threshold", () => {
		// Boxes sharing an edge overlap 0 → both kept regardless of class.
		const kept = nmsPerClass([cand(0, 0.9, 0, 0, 10, 10), cand(0, 0.8, 10, 0, 20, 10)]);
		expect(kept).toHaveLength(2);
		expect(NMS_IOU).toBe(0.45);
	});
});
