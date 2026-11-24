import { describe, expect, it } from "vitest";
import {
	captureFrame,
	hashDistance,
	MAX_FRAME_EDGE,
	scaledSize,
} from "../../../src/content/video/frames";
import { asVideo, FakeVideo, makeHarness } from "./helpers";

describe("scaledSize", () => {
	it("caps the long edge at MAX_FRAME_EDGE and keeps aspect ratio", () => {
		expect(scaledSize(640, 360)).toEqual({ width: MAX_FRAME_EDGE, height: 270 });
		expect(scaledSize(360, 640)).toEqual({ width: 270, height: MAX_FRAME_EDGE });
		expect(scaledSize(1280, 720)).toEqual({ width: 480, height: 270 });
	});
	it("does not upscale small videos", () => {
		expect(scaledSize(100, 50)).toEqual({ width: 100, height: 50 });
	});
});

describe("hashDistance", () => {
	it("is 0 for identical hashes and 1 for opposite bytes", () => {
		const a = new Uint8Array(256).fill(0);
		const b = new Uint8Array(256).fill(0);
		const c = new Uint8Array(256).fill(255);
		expect(hashDistance(a, b)).toBe(0);
		expect(hashDistance(a, c)).toBe(1);
	});
	it("scales with pixel difference", () => {
		const a = new Uint8Array(256).fill(0);
		const b = new Uint8Array(256).fill(128);
		expect(hashDistance(a, b)).toBeCloseTo(128 / 255, 2);
	});
});

describe("captureFrame", () => {
	it("returns null when the video has no data", async () => {
		const h = makeHarness();
		const v = new FakeVideo();
		v.readyState = 1;
		expect(await captureFrame(asVideo(v), h.env)).toBeNull();
	});

	it("returns null for zero-size video", async () => {
		const h = makeHarness();
		const v = new FakeVideo();
		v.videoWidth = 0;
		expect(await captureFrame(asVideo(v), h.env)).toBeNull();
	});

	it("exports a frame data URL and a hash", async () => {
		const h = makeHarness();
		const v = new FakeVideo();
		const frame = await captureFrame(asVideo(v), h.env);
		expect(frame).not.toBeNull();
		// The fake canvas can't encode WebP, so the PNG fallback path runs.
		expect(frame?.dataUrl.startsWith("data:image/")).toBe(true);
		expect(frame?.width).toBe(480);
		expect(frame?.height).toBe(270);
		expect(frame?.hash).toHaveLength(256);
		expect(frame?.ts).toBe(v.currentTime);
	});
});
