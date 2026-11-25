import { describe, expect, it } from "vitest";
import {
	associateFacesToPersons,
	boxContainment,
	composeMaskRgba,
	decodePersons,
	float16BitsToFloat32,
	float32ToFloat16Bits,
	type PersonInstance,
	personInputSizeFromShape,
	personLetterbox,
	tensorDataToFloat32,
} from "../../src/engine/person";
import { base64ToBytes, bytesToBase64, maskBitsToRgba, packAlphaBits } from "../../src/shared/mask";

function tensor(
	rows: number,
	cols: number,
): { data: Float32Array; set: (row: number, col: number, value: number) => void } {
	const data = new Float32Array(rows * cols);
	return {
		data,
		set(row, col, value) {
			data[row * cols + col] = value;
		},
	};
}

describe("person letterbox/input metadata", () => {
	it("uses centered Ultralytics padding", () => {
		expect(personLetterbox(800, 400, { width: 640, height: 640 })).toEqual({
			scale: 0.8,
			dw: 640,
			dh: 320,
			padX: 0,
			padY: 160,
			inputW: 640,
			inputH: 640,
		});
	});

	it("reads static HW input dims and falls back to 640", () => {
		expect(personInputSizeFromShape([1, 3, 512, 768])).toEqual({ width: 768, height: 512 });
		expect(personInputSizeFromShape(["batch", 3, "height", "width"])).toEqual({
			width: 640,
			height: 640,
		});
		expect(personInputSizeFromShape(undefined)).toEqual({ width: 640, height: 640 });
	});
});

describe("decodePersons", () => {
	it("decodes person boxes and coefficient rows using tensor-derived class counts", () => {
		// 1-class person fine-tune + 3 mask coefficients → 4+1+3 rows.
		const rows = 8;
		const cols = 2;
		const t = tensor(rows, cols);
		// Candidate 0: centered at input (320,320), 200x100.
		t.set(0, 0, 320);
		t.set(1, 0, 320);
		t.set(2, 0, 200);
		t.set(3, 0, 100);
		t.set(4, 0, 0.9);
		t.set(5, 0, 2);
		t.set(6, 0, -1);
		t.set(7, 0, 0.5);
		// Candidate 1 is below the score floor.
		t.set(4, 1, 0.1);

		const lb = personLetterbox(800, 400, { width: 640, height: 640 });
		const out = decodePersons({
			detData: t.data,
			detRows: rows,
			detCols: cols,
			protoDims: [3, 160, 160],
			letterbox: lb,
			srcW: 800,
			srcH: 400,
		});
		expect(out).toHaveLength(1);
		expect(out[0]!.score).toBeCloseTo(0.9);
		// Input xyxy [220,270]-[420,370] → centered-letterbox-unmapped normalized xywh.
		expect(out[0]!.box[0]).toBeCloseTo(0.34375);
		expect(out[0]!.box[1]).toBeCloseTo(0.34375);
		expect(out[0]!.box[2]).toBeCloseTo(0.3125);
		expect(out[0]!.box[3]).toBeCloseTo(0.3125);
		expect([...out[0]!.coeffs]).toEqual([2, -1, 0.5]);
	});

	it("detects normalized xywh exports and scales them into input pixels", () => {
		const rows = 8;
		const cols = 1;
		const t = tensor(rows, cols);
		// 640-input normalized xywh: center (320,320), size 200x100.
		t.set(0, 0, 0.5);
		t.set(1, 0, 0.5);
		t.set(2, 0, 200 / 640);
		t.set(3, 0, 100 / 640);
		t.set(4, 0, 0.9);
		const out = decodePersons({
			detData: t.data,
			detRows: rows,
			detCols: cols,
			protoDims: [3, 160, 160],
			letterbox: personLetterbox(640, 640),
			srcW: 640,
			srcH: 640,
		});
		expect(out).toHaveLength(1);
		expect(out[0]!.inputBox[0]).toBeCloseTo(220);
		expect(out[0]!.inputBox[1]).toBeCloseTo(270);
	});

	it("suppresses overlapping duplicates with NMS", () => {
		// 2 classes + 2 mask coefficients → 8 rows; both candidates are persons.
		const rows = 8;
		const cols = 2;
		const t = tensor(rows, cols);
		for (const [col, score] of [
			[0, 0.9],
			[1, 0.8],
		] as const) {
			t.set(0, col, 320);
			t.set(1, col, 320);
			t.set(2, col, 100);
			t.set(3, col, 100);
			t.set(4, col, score); // person class 0
			t.set(5, col, 0.01); // other class stays low
		}
		const out = decodePersons({
			detData: t.data,
			detRows: rows,
			detCols: cols,
			protoDims: [2, 160, 160],
			letterbox: personLetterbox(640, 640),
			srcW: 640,
			srcH: 640,
		});
		expect(out).toHaveLength(1);
		expect(out[0]!.score).toBeCloseTo(0.9);
	});

	it("caps decoded instances after NMS", () => {
		const rows = 8;
		const cols = 6;
		const t = tensor(rows, cols);
		for (let c = 0; c < cols; c++) {
			t.set(0, c, 0.08 + c * 0.17);
			t.set(1, c, 0.5);
			t.set(2, c, 0.08);
			t.set(3, c, 0.2);
			t.set(4, c, 0.9 - c * 0.01);
		}
		const out = decodePersons({
			detData: t.data,
			detRows: rows,
			detCols: cols,
			protoDims: [3, 160, 160],
			letterbox: personLetterbox(640, 640),
			srcW: 640,
			srcH: 640,
		});
		expect(out).toHaveLength(4);
		expect(out.map((p) => p.score)).toHaveLength(4);
		expect(out[0]!.score).toBeCloseTo(0.9);
		expect(out[3]!.score).toBeCloseTo(0.87);
	});

	it("throws when rows cannot contain any class score", () => {
		expect(() =>
			decodePersons({
				detData: new Float32Array(36),
				detRows: 36,
				detCols: 1,
				protoDims: [32, 160, 160],
				letterbox: personLetterbox(100, 100),
				srcW: 100,
				srcH: 100,
			}),
		).toThrow(/unexpected segmentation rows/);
	});
});

describe("composeMaskRgba", () => {
	const instance = (coeffs: number[]): PersonInstance => ({
		score: 0.9,
		box: [0.25, 0.25, 0.5, 0.5],
		inputBox: [1, 1, 3, 3],
		coeffs: new Float32Array(coeffs),
	});

	it("matrix-multiplies coefficients and prototypes and thresholds positive logits", () => {
		const raster = composeMaskRgba(
			instance([2]),
			new Float32Array([1, 1, 1, 1]),
			[1, 2, 2],
			4,
			4,
			16,
		);
		expect(raster.width).toBeGreaterThan(0);
		expect(raster.height).toBeGreaterThan(0);
		for (let i = 3; i < raster.rgba.length; i += 4) expect(raster.rgba[i]).toBe(255);
	});

	it("leaves negative prototype logits transparent", () => {
		const raster = composeMaskRgba(
			instance([1]),
			new Float32Array([-1, -1, -1, -1]),
			[1, 2, 2],
			4,
			4,
			16,
		);
		for (let i = 3; i < raster.rgba.length; i += 4) expect(raster.rgba[i]).toBe(0);
	});
});

describe("face/person association", () => {
	const person = (box: [number, number, number, number], score = 0.9): PersonInstance => ({
		score,
		box,
		inputBox: [0, 0, 1, 1],
		coeffs: new Float32Array(32),
	});

	it("computes face containment", () => {
		expect(boxContainment([0.4, 0.4, 0.1, 0.1], [0, 0, 1, 1])).toBeCloseTo(1);
		expect(boxContainment([0.95, 0.4, 0.1, 0.1], [0, 0, 1, 1])).toBeCloseTo(0.5);
	});

	it("associates a contained face and lets multiple faces share one person", () => {
		const out = associateFacesToPersons(
			[
				[0.4, 0.1, 0.08, 0.08],
				[0.4, 0.35, 0.08, 0.08],
			],
			[person([0.2, 0, 0.5, 0.9])],
		);
		expect(out).toHaveLength(2);
		expect(out.map((a) => a.personIndex)).toEqual([0, 0]);
	});

	it("drops faces outside every person box beyond the weak-overlap fallback", () => {
		expect(associateFacesToPersons([[0.8, 0.8, 0.1, 0.1]], [person([0, 0, 0.5, 0.5])])).toEqual([]);
	});

	it("uses IoU as a fallback for a truncated containment overlap", () => {
		const out = associateFacesToPersons([[0.55, 0.55, 0.1, 0.1]], [person([0.4, 0.4, 0.2, 0.2])]);
		expect(out).toHaveLength(1);
		expect(out[0]!.containment).toBeLessThan(0.35);
		expect(out[0]!.iou).toBeGreaterThan(0.02);
	});

	it("breaks overlapping-person ties toward the tighter/higher-IoU instance", () => {
		const out = associateFacesToPersons(
			[[0.1, 0.1, 0.1, 0.1]],
			[person([0, 0, 1, 1], 0.99), person([0.05, 0.05, 0.2, 0.2], 0.8)],
		);
		expect(out[0]!.personIndex).toBe(1);
	});
});

describe("float16 tensor conversion", () => {
	it("round-trips common values through IEEE half precision", () => {
		expect(float32ToFloat16Bits(1)).toBe(0x3c00);
		expect(float32ToFloat16Bits(-2)).toBe(0xc000);
		expect(float32ToFloat16Bits(0.5)).toBe(0x3800);
		expect(float16BitsToFloat32(0x3c00)).toBe(1);
		expect(float16BitsToFloat32(0xc000)).toBe(-2);
		expect(float16BitsToFloat32(0x3800)).toBe(0.5);
		for (const v of [0.25, 1 / 255, 0.8956701755523682]) {
			expect(float16BitsToFloat32(float32ToFloat16Bits(v))).toBeCloseTo(v, 3);
		}
	});

	it("expands Uint16 bit-storage and copies float32 tensor data", () => {
		const half = tensorDataToFloat32(new Uint16Array([0x3c00, 0xc000, 0x3800]));
		expect([...half]).toEqual([1, -2, 0.5]);
		const f32 = tensorDataToFloat32(new Float32Array([1.25, -0.5]));
		expect([...f32]).toEqual([1.25, -0.5]);
	});

	it("copies numeric native Float16Array-like data without bit-decoding it", () => {
		const nativeLike = { length: 2, 0: 0.895, 1: -0.25 } as ArrayLike<number>;
		const converted = tensorDataToFloat32(nativeLike);
		expect(converted[0]).toBeCloseTo(0.895);
		expect(converted[1]).toBeCloseTo(-0.25);
	});
});

describe("mask bit codec", () => {
	it("packs alpha MSB-first and round-trips base64", () => {
		const rgba = new Uint8ClampedArray(10 * 4);
		rgba[3] = 255; // pixel 0 → 0x80 bit 0
		rgba[2 * 4 + 3] = 255; // pixel 2
		rgba[9 * 4 + 3] = 255; // pixel 9 → byte 1 bit 1
		const packed = packAlphaBits(rgba, 5, 2);
		expect(packed[0]).toBe(0b10100000);
		expect(packed[1]).toBe(0b01000000);
		const encoded = bytesToBase64(packed);
		expect([...base64ToBytes(encoded)]).toEqual([...packed]);
		const expanded = maskBitsToRgba(encoded, 5, 2);
		expect(expanded[3]).toBe(255);
		expect(expanded[2 * 4 + 3]).toBe(255);
		expect(expanded[9 * 4 + 3]).toBe(255);
		expect(expanded[1 * 4 + 3]).toBe(0);
	});
});
