import { describe, expect, it, vi } from "vitest";
import { createActor } from "xstate";
import {
	applyVerdict,
	BASE_FPS,
	BOOST_FPS,
	HYSTERESIS_WINDOW,
	isHighConfidence,
	type SamplerContext,
	type SamplerEvent,
	STABLE_FPS,
	samplerMachine,
	VideoSampler,
} from "../../../src/content/video/sampler";
import { asVideo, FakeVideo, makeHarness, verdict } from "./helpers";

const ctx = (decisions: SamplerContext["decisions"], blurred = false): SamplerContext => ({
	videoId: "v1",
	blurred,
	decisions,
	lastVerdict: null,
	stableStreak: 0,
});

describe("samplerMachine transitions", () => {
	const states = (events: SamplerEvent[]) => {
		const actor = createActor(samplerMachine, { input: { videoId: "v" } });
		actor.start();
		for (const e of events) actor.send(e);
		return actor.getSnapshot().value;
	};

	it("walks idle → playing → sampling → paused → playing → sampling → ended", () => {
		expect(states([{ type: "PLAYING" }])).toBe("playing");
		expect(states([{ type: "PLAYING" }, { type: "SLOT_GRANTED" }])).toBe("sampling");
		expect(
			states([
				{ type: "PLAYING" },
				{ type: "SLOT_GRANTED" },
				{ type: "PAUSE" },
				{ type: "PLAYING" },
				{ type: "SLOT_GRANTED" },
				{ type: "ENDED" },
			]),
		).toBe("ended");
	});

	it("goes hidden on DOC_HIDDEN and back to playing on DOC_VISIBLE", () => {
		expect(states([{ type: "PLAYING" }, { type: "SLOT_GRANTED" }, { type: "DOC_HIDDEN" }])).toBe(
			"hidden",
		);
		expect(
			states([
				{ type: "PLAYING" },
				{ type: "SLOT_GRANTED" },
				{ type: "DOC_HIDDEN" },
				{ type: "DOC_VISIBLE" },
			]),
		).toBe("playing");
	});

	it("goes offscreen and back on-screen", () => {
		expect(
			states([
				{ type: "PLAYING" },
				{ type: "SLOT_GRANTED" },
				{ type: "OFF_SCREEN" },
				{ type: "ON_SCREEN" },
			]),
		).toBe("playing");
	});

	it("waits in playing until a slot is granted", () => {
		expect(states([{ type: "PLAYING" }])).toBe("playing");
	});
});

describe("applyVerdict hysteresis", () => {
	it("does not blur on a single ordinary blur verdict", () => {
		const next = applyVerdict(ctx([]), verdict("blur", 0.8));
		expect(next.blurred).toBe(false);
	});

	it("blurs after two consecutive blur verdicts", () => {
		const next = applyVerdict(ctx(["blur"]), verdict("blur", 0.8));
		expect(next.blurred).toBe(true);
	});

	it("blurs immediately on one high-confidence blur verdict", () => {
		const next = applyVerdict(ctx([]), verdict("blur", 0.95));
		expect(next.blurred).toBe(true);
	});

	it("stays blurred through one stray low-confidence clear verdict", () => {
		const next = applyVerdict(ctx(["blur", "blur"], true), verdict("clear", 0.3, 1));
		expect(next.blurred).toBe(true);
	});

	it("unblurs after two consecutive clear verdicts", () => {
		const next = applyVerdict(ctx(["clear"], true), verdict("clear", 0.3, 1));
		expect(next.blurred).toBe(false);
	});

	it("unblurs immediately on a high-confidence clear verdict (no detections)", () => {
		const next = applyVerdict(ctx(["blur"], true), verdict("clear"));
		expect(next.blurred).toBe(false);
	});

	it("keeps only HYSTERESIS_WINDOW decisions", () => {
		const next = applyVerdict(ctx(["blur", "blur"]), verdict("clear"));
		expect(next.decisions).toHaveLength(HYSTERESIS_WINDOW);
	});
});

describe("isHighConfidence", () => {
	it("blur verdict ≥0.9 is high confidence", () => {
		expect(isHighConfidence(verdict("blur", 0.95))).toBe(true);
	});
	it("blur verdict <0.9 is not", () => {
		expect(isHighConfidence(verdict("blur", 0.8))).toBe(false);
	});
	it("clear with zero detections is high confidence", () => {
		expect(isHighConfidence(verdict("clear"))).toBe(true);
	});
	it("clear with leftover detections is not", () => {
		expect(isHighConfidence({ ...verdict("clear"), reasons: verdict("blur", 0.3).reasons })).toBe(
			false,
		);
	});
});

describe("VideoSampler pause rules", () => {
	const setup = (overrides: { paused?: boolean } = {}) => {
		const h = makeHarness();
		const video = new FakeVideo();
		video.paused = overrides.paused ?? false;
		const sampler = new VideoSampler(asVideo(video), "v1", {
			classifyFrame: h.classifyFrame,
			env: h.env,
		});
		sampler.start();
		return { h, video, sampler };
	};

	it("starts playing when the video is already playing", () => {
		const { sampler } = setup();
		expect(sampler.state).toBe("playing");
	});

	it("stays idle when the video is paused", () => {
		const { sampler } = setup({ paused: true });
		expect(sampler.state).toBe("idle");
	});

	it("samples only after a slot grant, then pauses on pause event", async () => {
		const { h, video, sampler } = setup();
		sampler.grantSlot();
		expect(sampler.state).toBe("sampling");
		h.pumpFrames();
		await vi.waitFor(() => expect(h.classifyFrame).toHaveBeenCalled());
		video.pause();
		expect(sampler.state).toBe("paused");
		const calls = h.classifyFrame.mock.calls.length;
		h.pumpFrames();
		expect(h.classifyFrame).toHaveBeenCalledTimes(calls);
	});

	it("does no work while the document is hidden", async () => {
		const { h, sampler } = setup();
		sampler.grantSlot();
		h.doc.setHidden(true);
		expect(sampler.state).toBe("hidden");
		const calls = h.classifyFrame.mock.calls.length;
		h.pumpFrames();
		expect(h.classifyFrame).toHaveBeenCalledTimes(calls);
		h.doc.setHidden(false);
		expect(sampler.state).toBe("playing");
	});

	it("does no work while offscreen", async () => {
		const { h, sampler } = setup();
		sampler.grantSlot();
		h.io.fire(false);
		expect(sampler.state).toBe("offscreen");
		h.pumpFrames();
		expect(h.classifyFrame).not.toHaveBeenCalled();
		h.io.fire(true);
		expect(sampler.state).toBe("playing");
	});

	it("reconciles a paused video mid-tick without a pause event", async () => {
		const { h, video, sampler } = setup();
		sampler.grantSlot();
		video.paused = true; // element paused without firing an event
		h.pumpFrames();
		expect(sampler.state).toBe("paused");
		expect(h.classifyFrame).not.toHaveBeenCalled();
	});

	it("ends on ended event and stops sampling", async () => {
		const { h, video, sampler } = setup();
		sampler.grantSlot();
		video.end();
		expect(sampler.state).toBe("ended");
		h.pumpFrames();
		expect(h.classifyFrame).not.toHaveBeenCalled();
	});

	it("keeps at most one classify request in flight", async () => {
		const h = makeHarness();
		let resolve: (v: ReturnType<typeof verdict>) => void = () => {};
		h.classifyFrame.mockImplementation(() => new Promise((r) => (resolve = r)));
		const video = new FakeVideo();
		video.paused = false;
		const sampler = new VideoSampler(asVideo(video), "v1", {
			classifyFrame: h.classifyFrame,
			env: h.env,
		});
		sampler.start();
		sampler.grantSlot();
		h.pumpFrames();
		await Promise.resolve();
		await Promise.resolve();
		expect(h.classifyFrame).toHaveBeenCalledTimes(1);
		h.advance(1000);
		video.currentTime += 0.5;
		h.pumpFrames();
		await Promise.resolve();
		expect(h.classifyFrame).toHaveBeenCalledTimes(1);
		resolve(verdict("clear"));
	});

	it("skips duplicate frames (unchanged currentTime)", async () => {
		const { h, sampler } = setup();
		sampler.grantSlot();
		h.pumpFrames();
		await vi.waitFor(() => expect(h.classifyFrame).toHaveBeenCalledTimes(1));
		// currentTime unchanged and identical pixels → duplicate → skip
		h.advance(600);
		h.pumpFrames();
		await Promise.resolve();
		expect(h.classifyFrame).toHaveBeenCalledTimes(1);
	});
});

describe("adaptive rate", () => {
	const setup = () => {
		const h = makeHarness();
		const video = new FakeVideo();
		video.paused = false;
		const sampler = new VideoSampler(asVideo(video), "v1", {
			classifyFrame: h.classifyFrame,
			env: h.env,
		});
		sampler.start();
		sampler.grantSlot();
		return { h, video, sampler };
	};

	const sampleOnce = async (h: ReturnType<typeof makeHarness>, video: FakeVideo, stepMs = 500) => {
		h.advance(stepMs);
		video.currentTime += stepMs / 1000;
		h.pumpFrames();
		await Promise.resolve();
		await Promise.resolve();
		await Promise.resolve();
	};

	it("baseline rate is 2 fps", async () => {
		const { sampler } = setup();
		expect(sampler.currentRateFps).toBe(BASE_FPS);
	});

	it("boosts to 6 fps after a scene change", async () => {
		const { h, video, sampler } = setup();
		await sampleOnce(h, video);
		// Force a big pixel diff: new frames filled with opposite luma.
		h.setFill(() => 250);
		await sampleOnce(h, video);
		expect(sampler.currentRateFps).toBe(BOOST_FPS);
	});

	it("backs off to 1 fps when verdicts are stable", async () => {
		const { h, video, sampler } = setup();
		for (let i = 0; i < 8; i++) await sampleOnce(h, video);
		expect(sampler.currentRateFps).toBe(STABLE_FPS);
	});
});
