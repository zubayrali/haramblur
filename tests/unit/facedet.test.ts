import { describe, expect, it } from "vitest";
import {
	blazeRgbaToTensor,
	boxIou,
	corroboratedFaces,
	FACE_CORROBORATE_MIN,
	FACE_DET_PASSES,
	FACE_DET_SIZE,
	mergeDetectionPasses,
	type NormBox,
	parseFaceBoxes,
	sameFace,
	suppressCoveredFaceLabels,
} from "../../src/engine/facedet";
import type { Detection } from "../../src/shared/protocol";

const det = (
	label: Detection["label"],
	score: number,
	box: [number, number, number, number] = [0.4, 0.3, 0.2, 0.2],
): Detection => ({ label, score, box });

/** Build a `selectedBoxes` row: [top_y, top_x, bot_y, bot_x, ...landmarks]. */
const row = (y0: number, x0: number, y1: number, x1: number): number[] => [
	y0,
	x0,
	y1,
	x1,
	...Array.from({ length: 12 }, () => 0.5),
];

describe("parseFaceBoxes", () => {
	it("converts top_y/top_x/bot_y/bot_x rows to normalized xywh", () => {
		const data = new Float32Array(row(0.2, 0.3, 0.5, 0.6));
		expect(parseFaceBoxes(data, [1, 1, 16])).toEqual([[0.3, 0.2, 0.3, 0.3]]);
	});

	it("handles the collapsed single-detection shape [1,16]", () => {
		const data = new Float32Array(row(0.1, 0.1, 0.4, 0.4));
		expect(parseFaceBoxes(data, [1, 16])).toHaveLength(1);
	});

	it("returns empty for the no-detection shape [1,0,16] and bad dims", () => {
		expect(parseFaceBoxes(new Float32Array(0), [1, 0, 16])).toEqual([]);
		expect(parseFaceBoxes(new Float32Array(8), [1, 8])).toEqual([]);
	});

	it("parses multiple rows and clamps out-of-range coords", () => {
		const data = new Float32Array([...row(0.1, 0.1, 0.3, 0.3), ...row(-0.1, 0.5, 1.2, 0.9)]);
		const boxes = parseFaceBoxes(data, [1, 2, 16]);
		expect(boxes).toHaveLength(2);
		// Second row: x clamped 0.5..0.9, y clamped 0..1
		expect(boxes[1]).toEqual([0.5, 0, 0.4, 1]);
	});

	it("drops degenerate boxes", () => {
		const data = new Float32Array(row(0.5, 0.5, 0.5, 0.9));
		expect(parseFaceBoxes(data, [1, 1, 16])).toEqual([]);
	});
});

describe("mergeDetectionPasses", () => {
	const p = FACE_DET_PASSES;

	it("earns the highest pass score a box survives", () => {
		const lowOnly: NormBox = [0.3, 0.3, 0.2, 0.2];
		const both: NormBox = [0.1, 0.1, 0.2, 0.2];
		const out = mergeDetectionPasses([[lowOnly, both], [both], [both]]);
		expect(out).toHaveLength(2);
		const upgraded = out.find((o) => o.box[0] === 0.1);
		const kept = out.find((o) => o.box[0] === 0.3);
		expect(upgraded?.score).toBe(p[2]!.score);
		expect(kept?.score).toBe(p[0]!.score);
	});

	it("keeps higher-pass boxes that have no lower-pass match", () => {
		const box: NormBox = [0.5, 0.5, 0.2, 0.2];
		const out = mergeDetectionPasses([[], [box], [box]]);
		expect(out).toEqual([{ box, score: p[2]!.score }]);
	});
});

describe("sameFace / boxIou", () => {
	it("matches overlapping boxes and center-contained boxes", () => {
		const tight: [number, number, number, number] = [0.4, 0.4, 0.1, 0.1];
		const loose: [number, number, number, number] = [0.3, 0.3, 0.3, 0.3];
		expect(boxIou(tight, loose)).toBeGreaterThan(0);
		expect(sameFace(tight, loose)).toBe(true);
		expect(sameFace([0.05, 0.05, 0.05, 0.05], [0.8, 0.8, 0.1, 0.1])).toBe(false);
	});
});

describe("corroboratedFaces", () => {
	const scored = [{ box: [0.3, 0.3, 0.2, 0.2] as NormBox, score: 0.85 }];

	it("keeps a detector box overlapped by a NudeNet FACE_* detection", () => {
		const dets = [det("FACE_FEMALE", 0.4, [0.32, 0.31, 0.18, 0.18])];
		expect(corroboratedFaces(scored, dets)).toHaveLength(1);
	});

	it("drops detector boxes with no overlapping NudeNet face (dog FP guard)", () => {
		const dets = [det("BELLY_EXPOSED", 0.6), det("FACE_MALE", 0.1, [0.8, 0.8, 0.1, 0.1])];
		// FACE_MALE@0.1 is below the corroboration floor; BELLY is not a face.
		expect(corroboratedFaces(scored, dets)).toHaveLength(0);
	});

	it("drops detector boxes when NudeNet found no faces at all", () => {
		expect(corroboratedFaces(scored, [])).toHaveLength(0);
	});

	it("respects the corroboration score floor", () => {
		const justBelow = det("FACE_FEMALE", FACE_CORROBORATE_MIN - 0.01, [0.32, 0.31, 0.18, 0.18]);
		expect(corroboratedFaces(scored, [justBelow])).toHaveLength(0);
	});
});

describe("suppressCoveredFaceLabels", () => {
	it("drops NudeNet FACE_* boxes covered by a detector box", () => {
		const dets = [
			det("FACE_FEMALE", 0.53, [0.28, 0.22, 0.55, 0.6]),
			det("FACE_MALE", 0.4, [0.05, 0.05, 0.1, 0.1]),
			det("BELLY_EXPOSED", 0.61),
		];
		const out = suppressCoveredFaceLabels(dets, [[0.25, 0.2, 0.6, 0.62]]);
		// First face overlaps → dropped; second face + body detection survive.
		expect(out.map((d) => d.label)).toEqual(["FACE_MALE", "BELLY_EXPOSED"]);
	});

	it("passes everything through when no faces were detected", () => {
		const dets = [det("FACE_FEMALE", 0.9)];
		expect(suppressCoveredFaceLabels(dets, [])).toEqual(dets);
	});
});

describe("blazeRgbaToTensor", () => {
	it("produces CHW float32 RGB in 0..1", () => {
		const rgba = new Uint8ClampedArray(FACE_DET_SIZE * FACE_DET_SIZE * 4);
		// Pixel 0: pure red; pixel 1: pure green.
		rgba[0] = 255;
		rgba[5] = 255;
		const t = blazeRgbaToTensor(rgba);
		const plane = FACE_DET_SIZE * FACE_DET_SIZE;
		expect(t[0]).toBe(1); // R of pixel 0
		expect(t[plane + 1]).toBe(1); // G of pixel 1
		expect(t.length).toBe(3 * plane);
	});
});
