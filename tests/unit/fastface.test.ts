import { describe, expect, it } from "vitest";
import {
	FACE_INPUT_SIZE,
	FACE_MAX_PER_PASS,
	FACE_MIN_PX,
	FACE_RELABEL_MIN_CONF,
	faceAlignTransform,
	faceCropRect,
	faceRgbaToTensor,
	fastfaceInputSizeFromShape,
	genderProbs,
	refineFaceDetections,
} from "../../src/engine/fastface";
import type { Detection } from "../../src/shared/protocol";

const det = (
	label: Detection["label"],
	score: number,
	box: [number, number, number, number] = [0.4, 0.3, 0.2, 0.2],
): Detection => ({ label, score, box });

const classifier =
	(femaleProb: number, maleProb = 1 - femaleProb) =>
	async () => ({ femaleProb, maleProb, age: 30 });

const SRC_W = 500;
const SRC_H = 400;

describe("refineFaceDetections", () => {
	it("relabels a confident opposite-gender call, keeping NudeNet's score", async () => {
		// The man-bw-portrait failure mode: NudeNet says female, FastFace says male.
		const out = await refineFaceDetections(
			[det("FACE_FEMALE", 0.83)],
			classifier(0.02, 0.98),
			SRC_W,
			SRC_H,
		);
		expect(out).toEqual([{ ...det("FACE_MALE", 0.83), age: 30 }]);
	});

	it("keeps NudeNet's label when FastFace agrees", async () => {
		const out = await refineFaceDetections(
			[det("FACE_FEMALE", 0.8)],
			classifier(0.97, 0.03),
			SRC_W,
			SRC_H,
		);
		expect(out).toEqual([{ ...det("FACE_FEMALE", 0.8), age: 30 }]);
	});

	it("collapses to neutral FACE_PERSON whenever FastFace is not confident", async () => {
		// The male→female FP channel: any sub-RELABEL confidence — weak lean
		// or near coin-flip — means the face is real but its gender isn't
		// verified, so NudeNet's gendered label can't stand either.
		const weak = classifier(FACE_RELABEL_MIN_CONF - 0.01);
		const out = await refineFaceDetections([det("FACE_FEMALE", 0.8)], weak, SRC_W, SRC_H);
		expect(out).toEqual([{ ...det("FACE_PERSON", 0.8), age: 30 }]);
		const coinFlip = classifier(0.5);
		const out2 = await refineFaceDetections(
			[det("FACE_FEMALE", 0.5), det("BELLY_EXPOSED", 0.5)],
			coinFlip,
			SRC_W,
			SRC_H,
		);
		expect(out2).toEqual([{ ...det("FACE_PERSON", 0.5), age: 30 }, det("BELLY_EXPOSED", 0.5)]);
	});

	it("collapses dual-gender labels on one face into a single corrected detection", async () => {
		// nude-david pattern: both labels fire on the same box.
		const out = await refineFaceDetections(
			[
				det("FACE_MALE", 0.29, [0.39, 0.11, 0.11, 0.08]),
				det("FACE_FEMALE", 0.22, [0.39, 0.11, 0.11, 0.08]),
			],
			classifier(0.99, 0.01),
			SRC_W,
			SRC_H,
		);
		expect(out).toEqual([{ ...det("FACE_FEMALE", 0.29, [0.39, 0.11, 0.11, 0.08]), age: 30 }]);
	});

	it("does not merge overlapping-but-distinct faces", async () => {
		const out = await refineFaceDetections(
			[det("FACE_FEMALE", 0.8, [0.1, 0.1, 0.1, 0.1]), det("FACE_MALE", 0.7, [0.5, 0.5, 0.1, 0.1])],
			classifier(0.5),
			SRC_W,
			SRC_H,
		);
		expect(out).toHaveLength(2);
	});

	it("skips boxes below FACE_MIN_PX without calling the classifier", async () => {
		let calls = 0;
		const spy = async () => {
			calls++;
			return { femaleProb: 0.99, maleProb: 0.01, age: 30 };
		};
		// 0.01 × 500px = 5px wide — below the floor.
		const out = await refineFaceDetections(
			[det("FACE_FEMALE", 0.9, [0.4, 0.3, 0.01, 0.05])],
			spy,
			SRC_W,
			SRC_H,
		);
		expect(calls).toBe(0);
		expect(out).toEqual([det("FACE_FEMALE", 0.9, [0.4, 0.3, 0.01, 0.05])]);
		expect(FACE_MIN_PX).toBeGreaterThan(0);
	});

	it("caps classifier runs at FACE_MAX_PER_PASS, highest scores first", async () => {
		// Well-separated boxes so none cluster into the same face.
		const boxes = Array.from({ length: FACE_MAX_PER_PASS + 3 }, (_, i) =>
			det("FACE_FEMALE", 0.9 - i * 0.01, [(i % 4) * 0.25, Math.floor(i / 4) * 0.3, 0.1, 0.1]),
		);
		let calls = 0;
		const spy = async () => {
			calls++;
			return { femaleProb: 0.5, maleProb: 0.5, age: 30 };
		};
		await refineFaceDetections(boxes, spy, SRC_W, SRC_H);
		expect(calls).toBe(FACE_MAX_PER_PASS);
	});

	it("leaves non-face detections untouched", async () => {
		const out = await refineFaceDetections(
			[det("FEMALE_BREAST_EXPOSED", 0.7), det("FACE_MALE", 0.6)],
			classifier(0.01, 0.99),
			SRC_W,
			SRC_H,
		);
		expect(out[0]).toEqual(det("FEMALE_BREAST_EXPOSED", 0.7));
	});

	it("keeps original detections when the classifier throws", async () => {
		const input = [det("FACE_FEMALE", 0.8)];
		const out = await refineFaceDetections(
			input,
			async () => {
				throw new Error("boom");
			},
			SRC_W,
			SRC_H,
		);
		expect(out).toEqual(input);
	});
});

describe("faceCropRect", () => {
	it("expands the box by the margin and clamps to the image", () => {
		// box [0.4,0.3,0.2,0.2] on 500×400 → px [200,120,100,80], margin 25%.
		const r = faceCropRect([0.4, 0.3, 0.2, 0.2], SRC_W, SRC_H);
		expect(r).not.toBeNull();
		// x: (0.4-0.05)*500=175, w grows by 2*0.05*500=50 → 150 wide? w=0.2*500=100 +50 =150
		expect(r!.x).toBeCloseTo(175);
		expect(r!.y).toBeCloseTo(100); // (0.3-0.05)*400
		expect(r!.w).toBeCloseTo(150);
		expect(r!.h).toBeCloseTo(120);
	});

	it("clamps boxes hanging off the image edge", () => {
		const r = faceCropRect([0.9, 0.9, 0.2, 0.2], SRC_W, SRC_H);
		expect(r!.x + r!.w).toBeLessThanOrEqual(SRC_W);
		expect(r!.y + r!.h).toBeLessThanOrEqual(SRC_H);
	});
});

describe("faceRgbaToTensor", () => {
	it("produces NCHW float32 with ImageNet normalization", () => {
		const rgba = new Uint8ClampedArray(112 * 112 * 4).fill(255);
		const t = faceRgbaToTensor(rgba);
		expect(t.length).toBe(3 * 112 * 112);
		// White pixel → (1 - mean) / std per channel.
		expect(t[0]).toBeCloseTo((1 - 0.485) / 0.229, 5);
		expect(t[112 * 112]).toBeCloseTo((1 - 0.456) / 0.224, 5);
		expect(t[2 * 112 * 112]).toBeCloseTo((1 - 0.406) / 0.225, 5);
	});
});

describe("genderProbs", () => {
	it("softmaxes the two logits, index 0 = female", () => {
		const { female, male } = genderProbs([2, 0]);
		expect(female).toBeCloseTo(0.881, 3);
		expect(male).toBeCloseTo(0.119, 3);
	});
});

describe("faceAlignTransform", () => {
	const apply = (
		t: { a: number; b: number; c: number; d: number; e: number; f: number },
		x: number,
		y: number,
	) => ({ x: t.a * x + t.c * y + t.e, y: t.b * x + t.d * y + t.f });

	it("returns null without two eye landmarks", () => {
		expect(faceAlignTransform([], 500, 400)).toBeNull();
		expect(faceAlignTransform([[0.5, 0.5]], 500, 400)).toBeNull();
	});

	it("maps the eye midpoint to (56, 39.2) and levels tilted eyes", () => {
		// Level eyes 50px apart on a 500×400 source, normalized coords.
		const t = faceAlignTransform(
			[
				[0.4, 0.3],
				[0.5, 0.3],
				[0.45, 0.35],
				[0.47, 0.4],
				[0.43, 0.4],
			],
			500,
			400,
		)!;
		const mid = apply(t, 225, 120); // eye midpoint in px
		expect(mid.x).toBeCloseTo(56, 3);
		expect(mid.y).toBeCloseTo(39.2, 3);
		// Inter-eye distance becomes 0.32 * 112 = 35.84.
		const re = apply(t, 200, 120);
		const le = apply(t, 250, 120);
		expect(Math.hypot(le.x - re.x, le.y - re.y)).toBeCloseTo(35.84, 3);
	});

	it("rotates a tilted eye-line horizontal", () => {
		const t = faceAlignTransform(
			[
				[0.4, 0.28],
				[0.5, 0.32], // eyes slant down-right
				[0.45, 0.35],
				[0.47, 0.4],
				[0.43, 0.4],
			],
			500,
			400,
		)!;
		const re = apply(t, 200, 112);
		const le = apply(t, 250, 128);
		// After alignment both eyes share the same y.
		expect(re.y).toBeCloseTo(le.y, 3);
	});
});

describe("fastfaceInputSizeFromShape", () => {
	it("reads the square edge off a valid NCHW shape", () => {
		expect(fastfaceInputSizeFromShape(["batch", 3, 112, 112])).toBe(112);
		expect(fastfaceInputSizeFromShape(["batch", 3, 128, 128])).toBe(128);
		expect(fastfaceInputSizeFromShape([1, 3, 160, 160])).toBe(160);
	});

	it("falls back to 112 for missing or malformed shapes", () => {
		expect(fastfaceInputSizeFromShape(undefined)).toBe(FACE_INPUT_SIZE);
		expect(fastfaceInputSizeFromShape([])).toBe(FACE_INPUT_SIZE);
		expect(fastfaceInputSizeFromShape([1, 3, 112])).toBe(FACE_INPUT_SIZE);
		expect(fastfaceInputSizeFromShape(["batch", 3, "H", "W"])).toBe(FACE_INPUT_SIZE);
	});

	it("rejects non-square and implausible sizes", () => {
		expect(fastfaceInputSizeFromShape([1, 3, 112, 128])).toBe(FACE_INPUT_SIZE);
		expect(fastfaceInputSizeFromShape([1, 3, 8, 8])).toBe(FACE_INPUT_SIZE);
		expect(fastfaceInputSizeFromShape([1, 3, 1024, 1024])).toBe(FACE_INPUT_SIZE);
	});
});

describe("faceRgbaToTensor with a non-default size", () => {
	it("produces 3·size² values laid out channel-planar", () => {
		const size = 128;
		const rgba = new Uint8ClampedArray(size * size * 4).fill(255);
		const tensor = faceRgbaToTensor(rgba, size);
		expect(tensor.length).toBe(3 * size * size);
	});
});
