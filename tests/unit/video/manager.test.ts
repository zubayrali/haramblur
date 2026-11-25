import { describe, expect, it, vi } from "vitest";
import { MAX_CONCURRENT_VIDEOS, VideoManager } from "../../../src/content/video/manager";
import type { Settings } from "../../../src/shared/settings";
import { DEFAULT_SETTINGS } from "../../../src/shared/settings";
import { asVideo, FakeVideo, makeHarness, verdict } from "./helpers";

async function makeManager(settings: Partial<Settings> = {}): Promise<{
	manager: VideoManager;
	h: ReturnType<typeof makeHarness>;
	settingsRef: Settings;
	watch: (next: Settings, prev: Settings) => void;
}> {
	const h = makeHarness();
	const settingsRef = { ...DEFAULT_SETTINGS, ...settings };
	let watcher: (next: Settings, prev: Settings) => void = () => {};
	const manager = new VideoManager({
		classifyFrame: h.classifyFrame,
		getSettings: async () => settingsRef,
		watchSettings: (cb) => {
			watcher = cb;
			return () => {};
		},
		env: h.env,
	});
	await manager.start();
	return { manager, h, settingsRef, watch: watcher };
}

describe("VideoManager", () => {
	it("samples at most MAX_CONCURRENT_VIDEOS videos at once", async () => {
		const { manager } = await makeManager();
		const videos = Array.from({ length: MAX_CONCURRENT_VIDEOS + 2 }, () => {
			const v = new FakeVideo();
			v.paused = false;
			return v;
		});
		for (const v of videos) manager.observe(asVideo(v));
		const states = videos.map((v) => manager.samplerFor(asVideo(v))?.state);
		expect(states.filter((s) => s === "sampling")).toHaveLength(MAX_CONCURRENT_VIDEOS);
		expect(states.filter((s) => s === "playing")).toHaveLength(2);
	});

	it("frees a slot when a sampled video pauses", async () => {
		const { manager } = await makeManager();
		const videos = Array.from({ length: MAX_CONCURRENT_VIDEOS + 1 }, () => {
			const v = new FakeVideo();
			v.paused = false;
			return v;
		});
		for (const v of videos) manager.observe(asVideo(v));
		const first = videos[0];
		const next = videos[MAX_CONCURRENT_VIDEOS];
		if (!first || !next) throw new Error("unreachable");
		first.pause();
		expect(manager.samplerFor(asVideo(first))?.state).toBe("paused");
		expect(manager.samplerFor(asVideo(next))?.state).toBe("sampling");
	});

	it("does nothing when blurVideos is off", async () => {
		const { manager } = await makeManager({ blurVideos: false });
		const v = new FakeVideo();
		v.paused = false;
		manager.observe(asVideo(v));
		expect(manager.size).toBe(0);
	});

	it("does nothing when status is off", async () => {
		const { manager } = await makeManager({ status: false });
		const v = new FakeVideo();
		v.paused = false;
		manager.observe(asVideo(v));
		expect(manager.size).toBe(0);
	});

	it("skips allowlisted hostnames", async () => {
		const { manager } = await makeManager({ allowlist: ["example.com"] });
		const v = new FakeVideo();
		v.paused = false;
		manager.observe(asVideo(v));
		expect(manager.size).toBe(0);
	});

	it("blurs while pending only when blurryStartMode is on", async () => {
		const { manager } = await makeManager({ blurryStartMode: true });
		const v = new FakeVideo();
		v.paused = false;
		manager.observe(asVideo(v));
		expect(v.dataset.hbState).toBe("pending");
		expect(v.dataset.hbBlur).toBe("1");
	});

	it("does not blur while pending when blurryStartMode is off", async () => {
		const { manager } = await makeManager({ blurryStartMode: false });
		const v = new FakeVideo();
		v.paused = false;
		manager.observe(asVideo(v));
		expect(v.dataset.hbState).toBe("pending");
		expect(v.dataset.hbBlur).toBeUndefined();
	});

	it("applies blur after two agreeing blur verdicts", async () => {
		const { manager, h } = await makeManager();
		h.classifyFrame.mockResolvedValue(verdict("blur", 0.8));
		const v = new FakeVideo();
		v.paused = false;
		manager.observe(asVideo(v));
		// First sample
		h.pumpFrames();
		await vi.waitFor(() => expect(h.classifyFrame).toHaveBeenCalledTimes(1));
		expect(v.dataset.hbState).not.toBe("blur");
		// Second sample (agreeing verdict → flip)
		h.advance(600);
		v.currentTime += 0.5;
		h.pumpFrames();
		await vi.waitFor(() => expect(v.dataset.hbState).toBe("blur"));
		expect(v.dataset.hbBlur).toBe("1");
		expect(v.dataset.hbReasons).toContain("FACE_FEMALE");
	});

	it("keeps blur on classify error when failClosed", async () => {
		const { manager, h } = await makeManager({ failClosed: true });
		h.classifyFrame.mockRejectedValue(new Error("engine down"));
		const v = new FakeVideo();
		v.paused = false;
		manager.observe(asVideo(v));
		h.pumpFrames();
		await vi.waitFor(() => expect(v.dataset.hbState).toBe("error"));
		expect(v.dataset.hbBlur).toBe("1");
	});

	it("unblurs but keeps error state on classify error when failClosed is off", async () => {
		const { manager, h } = await makeManager({ failClosed: false });
		h.classifyFrame.mockRejectedValue(new Error("engine down"));
		const v = new FakeVideo();
		v.paused = false;
		manager.observe(asVideo(v));
		h.pumpFrames();
		await vi.waitFor(() => expect(h.classifyFrame).toHaveBeenCalled());
		expect(v.dataset.hbState).toBe("error");
		expect(v.dataset.hbBlur).toBeUndefined();
	});

	it("reveals a blurred video on click and stays revealed", async () => {
		const { manager, h } = await makeManager({ revealModeVideos: "hover" });
		h.classifyFrame.mockResolvedValue(
			verdict("blur", 0.95, 1, "FEMALE_BREAST_EXPOSED"), // high confidence → instant blur
		);
		const v = new FakeVideo();
		v.paused = false;
		manager.observe(asVideo(v));
		h.pumpFrames();
		await vi.waitFor(() => expect(v.dataset.hbState).toBe("blur"));
		manager.reveal(asVideo(v));
		expect(v.dataset.hbBlur).toBeUndefined();
		expect(v.dataset.hbReveal).toBe("click");
		// Later verdicts must not re-blur a revealed video.
		h.advance(600);
		v.currentTime += 0.5;
		h.pumpFrames();
		await vi.waitFor(() => expect(h.classifyFrame).toHaveBeenCalledTimes(2));
		expect(v.dataset.hbBlur).toBeUndefined();
	});

	it("rebuilds blur from lastVerdict on settings change without wiping reasons", async () => {
		const { manager, h, settingsRef, watch } = await makeManager();
		h.classifyFrame.mockResolvedValue(verdict("blur", 0.95));
		const v = new FakeVideo();
		v.paused = false;
		manager.observe(asVideo(v));
		h.pumpFrames();
		await vi.waitFor(() => expect(h.classifyFrame).toHaveBeenCalledTimes(1));
		// Face-only verdicts need the second agreeing frame (no instant flip).
		h.advance(600);
		v.currentTime += 0.5;
		h.pumpFrames();
		await vi.waitFor(() => expect(v.dataset.hbState).toBe("blur"));
		watch({ ...settingsRef, blurAmount: 40 }, settingsRef);
		expect(v.dataset.hbState).toBe("blur");
		expect(v.dataset.hbBlur).toBe("1");
		expect(v.dataset.hbReasons).toContain("FACE_FEMALE");
	});

	it("stops and unblurs all videos when settings turn detection off", async () => {
		const { manager, h, settingsRef, watch } = await makeManager();
		h.classifyFrame.mockResolvedValue(verdict("blur", 0.95, 1, "FEMALE_BREAST_EXPOSED"));
		const v = new FakeVideo();
		v.paused = false;
		manager.observe(asVideo(v));
		h.pumpFrames();
		await vi.waitFor(() => expect(v.dataset.hbState).toBe("blur"));
		watch({ ...settingsRef, status: false }, settingsRef);
		expect(manager.size).toBe(0);
		expect(v.dataset.hbState).toBe("clear");
		expect(v.dataset.hbBlur).toBeUndefined();
	});

	it("unobserve disposes the sampler and clears blur", async () => {
		const { manager, h } = await makeManager();
		h.classifyFrame.mockResolvedValue(verdict("blur", 0.95, 1, "FEMALE_BREAST_EXPOSED"));
		const v = new FakeVideo();
		v.paused = false;
		manager.observe(asVideo(v));
		h.pumpFrames();
		await vi.waitFor(() => expect(v.dataset.hbState).toBe("blur"));
		manager.unobserve(asVideo(v));
		expect(manager.size).toBe(0);
		expect(v.dataset.hbState).toBe("clear");
	});
});
