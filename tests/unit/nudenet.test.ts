import { describe, expect, it } from "vitest";
import {
	defaultInputSizeForBackend,
	INPUT_SIZE,
	inputSizeOrDefault,
	LETTERBOX_FILL,
	letterboxSize,
	MAX_TILES_PER_AXIS,
	mergeDetections,
	NMS_IOU,
	nmsPerClass,
	parseDetections,
	resolveNudenetInputSize,
	rgbaToTensor,
	SCORE_FLOOR,
	staticInputSizeFromShape,
	tileRects,
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

describe("input size metadata", () => {
	it("accepts stride-aligned sizes in the supported range", () => {
		expect(inputSizeOrDefault(320, INPUT_SIZE)).toBe(320);
		expect(inputSizeOrDefault(416, INPUT_SIZE)).toBe(416);
		expect(inputSizeOrDefault(480, INPUT_SIZE)).toBe(480);
		expect(inputSizeOrDefault(640, INPUT_SIZE)).toBe(640);
	});

	it("rejects invalid or non-stride-aligned sizes", () => {
		for (const bad of [0, 255, 321, 1025, Number.NaN, -320]) {
			expect(inputSizeOrDefault(bad, INPUT_SIZE)).toBe(INPUT_SIZE);
		}
	});

	it("reads a fixed square input size from static ONNX metadata", () => {
		expect(staticInputSizeFromShape([1, 3, 640, 640])).toBe(640);
		expect(staticInputSizeFromShape(["batch", 3, 416, 416])).toBe(416);
		expect(staticInputSizeFromShape(["batch", 3, "height", "width"])).toBeNull();
		expect(staticInputSizeFromShape([1, 3, 416, 480])).toBeNull();
	});

	it("keeps WASM at 320 while allowing the GPU path more resolution", () => {
		expect(defaultInputSizeForBackend("wasm")).toBe(320);
		expect(defaultInputSizeForBackend("webgpu")).toBeGreaterThan(320);
	});

	it("resolves static graph, measurement override, pack metadata, then backend default", () => {
		expect(resolveNudenetInputSize({ backend: "webgpu", staticSize: 640, requestSize: 416 })).toBe(
			640,
		);
		expect(resolveNudenetInputSize({ backend: "webgpu", modelInputSize: 640 })).toBe(640);
		expect(
			resolveNudenetInputSize({ backend: "webgpu", modelInputSize: 640, requestSize: 416 }),
		).toBe(416);
		expect(resolveNudenetInputSize({ backend: "wasm" })).toBe(320);
	});
});

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

describe("tileRects", () => {
	it("returns nothing only when the source fits the input well under density", () => {
		// ≤ ~213px sources have no zoom to gain: ceil(213 × 1.5 / 320) = 1.
		expect(tileRects(200, 160)).toEqual([]);
		// Exactly-input sources still tile: a small face gains ~1.33× zoom.
		expect(tileRects(INPUT_SIZE, INPUT_SIZE)).toHaveLength(4);
	});

	it("covers each axis fully with overlapping tiles, capped per axis", () => {
		// 960×640 → 3×3 tiles; union covers [0,960]×[0,640] with overlap.
		const rects = tileRects(960, 640);
		expect(rects).toHaveLength(9);
		const xs = [...new Set(rects.map((r) => r.x))].sort((a, b) => a - b);
		const ys = [...new Set(rects.map((r) => r.y))].sort((a, b) => a - b);
		expect(xs[0]).toBe(0);
		expect(ys[0]).toBe(0);
		for (const [i, x] of xs.entries().filter(([, x]) => x > 0)) {
			// Each next column starts before the previous ends → overlap.
			const prevEnd =
				rects.find((r) => r.x === xs[i - 1])!.x + rects.find((r) => r.x === xs[i - 1])!.w;
			expect(x).toBeLessThan(prevEnd);
		}
		expect(Math.max(...rects.map((r) => r.x + r.w))).toBe(960);
		expect(Math.max(...rects.map((r) => r.y + r.h))).toBe(640);
	});

	it("never exceeds MAX_TILES_PER_AXIS² tiles even on huge sources", () => {
		const rects = tileRects(960, 4096);
		expect(rects.length).toBeLessThanOrEqual(MAX_TILES_PER_AXIS ** 2);
		// Tall source: 320×960 → 2×3 tiles at 1.5× density.
		expect(tileRects(320, 960)).toHaveLength(6);
	});
});

describe("mergeDetections", () => {
	const det = (label: string, score: number, x: number, w: number) => ({
		label: label as (typeof NUDENET_LABELS)[number],
		score,
		box: [x, 0, w, 0.2] as [number, number, number, number],
	});

	it("dedupes the same face seen across a tile seam, keeping the higher score", () => {
		const merged = mergeDetections(
			[det("FACE_FEMALE", 0.9, 0.4, 0.1)],
			[det("FACE_FEMALE", 0.7, 0.42, 0.1)],
		);
		expect(merged).toHaveLength(1);
		expect(merged[0]!.score).toBeCloseTo(0.9);
	});

	it("keeps disjoint same-class boxes and different classes; sorts by score", () => {
		const merged = mergeDetections(
			[det("FACE_FEMALE", 0.3, 0.0, 0.1)],
			[det("FACE_FEMALE", 0.8, 0.6, 0.1), det("FACE_MALE", 0.5, 0.0, 0.1)],
		);
		expect(merged.map((d) => d.score)).toEqual([0.8, 0.5, 0.3]);
		expect(merged.map((d) => d.label)).toContain("FACE_MALE");
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
