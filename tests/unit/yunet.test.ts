import { describe, expect, it } from "vitest";
import {
	decodeYuNet,
	landmarksPlausible,
	letterboxRect,
	unletterboxBox,
	unletterboxPoint,
	YUNET_INPUT_SIZE,
	type YuNetFace,
	type YuNetOutputs,
} from "../../src/engine/yunet";

/** Zero-filled outputs; tests poke individual anchors. */
function emptyOutputs(): YuNetOutputs {
	return {
		cls_8: new Float32Array(80 * 80),
		cls_16: new Float32Array(40 * 40),
		cls_32: new Float32Array(20 * 20),
		obj_8: new Float32Array(80 * 80),
		obj_16: new Float32Array(40 * 40),
		obj_32: new Float32Array(20 * 20),
		bbox_8: new Float32Array(80 * 80 * 4),
		bbox_16: new Float32Array(40 * 40 * 4),
		bbox_32: new Float32Array(20 * 20 * 4),
		kps_8: new Float32Array(80 * 80 * 10),
		kps_16: new Float32Array(40 * 40 * 10),
		kps_32: new Float32Array(20 * 20 * 10),
	};
}

/** Put a stride-32 detection at grid cell (r,c) with the given box offsets. */
function plantFace(
	outputs: YuNetOutputs,
	r: number,
	c: number,
	opts: { score?: number; dx?: number; dy?: number; w?: number; h?: number } = {},
): number {
	const cols = 20; // stride 32 on 640 input
	const idx = r * cols + c;
	outputs.cls_32[idx] = opts.score ?? 0.9;
	outputs.obj_32[idx] = opts.score ?? 0.9;
	const b = idx * 4;
	outputs.bbox_32[b] = opts.dx ?? 0.5;
	outputs.bbox_32[b + 1] = opts.dy ?? 0.5;
	outputs.bbox_32[b + 2] = Math.log(opts.w ?? 2);
	outputs.bbox_32[b + 3] = Math.log(opts.h ?? 2);
	// Landmarks: symmetric around the box center, eyes above mouth.
	const k = idx * 10;
	const cx = (c + (opts.dx ?? 0.5)) * 32;
	const cy = (r + (opts.dy ?? 0.5)) * 32;
	const w = (opts.w ?? 2) * 32;
	const h = (opts.h ?? 2) * 32;
	// offsets relative to cell (c,r), divided by stride
	const pts: [number, number][] = [
		[cx - w * 0.2 - c * 32, cy - h * 0.15 - r * 32], // right eye
		[cx + w * 0.2 - c * 32, cy - h * 0.15 - r * 32], // left eye
		[cx - c * 32, cy - r * 32], // nose
		[cx + w * 0.15 - c * 32, cy + h * 0.2 - r * 32], // mouth r
		[cx - w * 0.15 - c * 32, cy + h * 0.2 - r * 32], // mouth l
	];
	pts.forEach(([x, y], n) => {
		outputs.kps_32[k + 2 * n] = x / 32;
		outputs.kps_32[k + 2 * n + 1] = y / 32;
	});
	return idx;
}

describe("letterboxRect", () => {
	it("preserves aspect ratio and centers content", () => {
		const lb = letterboxRect(1600, 900); // 16:9
		expect(lb.scale).toBeCloseTo(640 / 1600);
		expect(lb.cw).toBeCloseTo(640);
		expect(lb.ch).toBeCloseTo(360);
		expect(lb.ox).toBe(0);
		expect(lb.oy).toBeCloseTo(140);
	});
	it("square sources fill the canvas", () => {
		const lb = letterboxRect(1000, 1000);
		expect(lb.cw).toBe(YUNET_INPUT_SIZE);
		expect(lb.ch).toBe(YUNET_INPUT_SIZE);
		expect(lb.ox).toBe(0);
		expect(lb.oy).toBe(0);
	});
});

describe("unletterbox", () => {
	it("round-trips normalized coords through the letterbox", () => {
		const lb = letterboxRect(1600, 900);
		const box = { x: 100, y: 200, w: 50, h: 80 };
		const n = unletterboxBox(box, lb, 1600, 900);
		// normalized → canvas px → normalized again should be identity-ish
		const back = {
			x: n.x * lb.cw + lb.ox,
			y: n.y * lb.ch + lb.oy,
			w: n.w * lb.cw,
			h: n.h * lb.ch,
		};
		expect(back.x).toBeCloseTo(box.x);
		expect(back.y).toBeCloseTo(box.y);
		expect(back.w).toBeCloseTo(box.w);
		expect(back.h).toBeCloseTo(box.h);
	});
	it("maps points", () => {
		const lb = letterboxRect(800, 800);
		expect(unletterboxPoint([320, 320], lb, 800, 800)).toEqual([0.5, 0.5]);
	});
});

describe("decodeYuNet", () => {
	it("decodes score = sqrt(cls*obj) and box/landmarks per OpenCV postProcess", () => {
		const o = emptyOutputs();
		plantFace(o, 10, 10, { score: 0.81, dx: 0.5, dy: 0.5, w: 2, h: 2 });
		const faces = decodeYuNet(o);
		expect(faces).toHaveLength(1);
		const f = faces[0]!;
		expect(f.score).toBeCloseTo(0.81, 5);
		// cx=(10+0.5)*32=336, cy=336, w=h=64 → x=304,y=304
		expect(f.box.x).toBeCloseTo(304);
		expect(f.box.y).toBeCloseTo(304);
		expect(f.box.w).toBeCloseTo(64);
		expect(f.box.h).toBeCloseTo(64);
		expect(f.landmarks).toHaveLength(5);
	});

	it("drops candidates below the score threshold", () => {
		const o = emptyOutputs();
		plantFace(o, 5, 5, { score: 0.5 }); // sqrt(0.5*0.5)=0.5 < 0.6
		expect(decodeYuNet(o)).toHaveLength(0);
	});

	it("NMS keeps the higher-scored of two overlapping faces", () => {
		const o = emptyOutputs();
		plantFace(o, 10, 10, { score: 0.9 });
		plantFace(o, 10, 11, { score: 0.7, dx: 0.5, dy: 0.5 }); // adjacent cell, overlapping box
		const faces = decodeYuNet(o, { nmsThreshold: 0.3 });
		expect(faces).toHaveLength(1);
		expect(faces[0]!.score).toBeCloseTo(0.9, 5);
	});

	it("keeps non-overlapping faces", () => {
		const o = emptyOutputs();
		plantFace(o, 2, 2, { score: 0.9 });
		plantFace(o, 17, 17, { score: 0.8 });
		expect(decodeYuNet(o)).toHaveLength(2);
	});
});

describe("landmarksPlausible", () => {
	const good: YuNetFace = {
		box: { x: 100, y: 100, w: 80, h: 100 },
		landmarks: [
			[130, 130], // right eye
			[170, 130], // left eye
			[150, 160], // nose
			[165, 185], // mouth r
			[135, 185], // mouth l
		],
		score: 0.9,
	};
	it("accepts a normal face", () => {
		expect(landmarksPlausible(good)).toBe(true);
	});
	it("rejects eyes below the mouth (inverted/snout geometry)", () => {
		const bad: YuNetFace = {
			...good,
			landmarks: [
				[130, 190],
				[170, 190],
				[150, 170],
				[165, 140],
				[135, 140],
			],
		};
		expect(landmarksPlausible(bad)).toBe(false);
	});
	it("rejects degenerate eye distance", () => {
		const bad: YuNetFace = {
			...good,
			landmarks: [
				[150, 130],
				[151, 130],
				[150, 160],
				[165, 185],
				[135, 185],
			],
		};
		expect(landmarksPlausible(bad)).toBe(false);
	});
	it("rejects landmarks far outside the box", () => {
		const bad: YuNetFace = {
			...good,
			landmarks: [
				[130, 130],
				[900, 130], // way outside
				[150, 160],
				[165, 185],
				[135, 185],
			],
		};
		expect(landmarksPlausible(bad)).toBe(false);
	});
});
