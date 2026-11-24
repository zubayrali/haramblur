import { describe, expect, it } from "vitest";
import { nsfwDetection, nsfwVerdict, softmax } from "../../src/engine/nsfw";

describe("softmax", () => {
	it("produces a probability distribution", () => {
		const p = softmax([1, 2, 3]);
		expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
		expect(p[2]!).toBeGreaterThan(p[0]!);
	});

	it("handles negative logits without overflow", () => {
		const p = softmax([-100, -90, -80]);
		expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
	});
});

describe("nsfwVerdict", () => {
	// Class order: Drawing, Hentai, Neutral, Porn, Sexy
	const probs = (drawing: number, hentai: number, neutral: number, porn: number, sexy: number) => [
		drawing,
		hentai,
		neutral,
		porn,
		sexy,
	];

	it("returns null when SFW wins", () => {
		expect(nsfwVerdict(probs(0.05, 0.05, 0.85, 0.03, 0.02))).toBeNull();
	});

	it("returns null when the margin is noise-level", () => {
		// Porn slightly ahead of Neutral → margin 0.01 < 0.02 floor.
		expect(nsfwVerdict(probs(0.0, 0.0, 0.495, 0.505, 0.0))).toBeNull();
	});

	it("labels explicit content when Porn/Hentai leads Sexy", () => {
		const v = nsfwVerdict(probs(0.02, 0.1, 0.2, 0.6, 0.08));
		expect(v?.label).toBe("NSFW_EXPLICIT");
		expect(v?.margin).toBeCloseTo(0.6 - 0.2);
	});

	it("labels suggestive content when Sexy leads", () => {
		const v = nsfwVerdict(probs(0.02, 0.05, 0.2, 0.1, 0.63));
		expect(v?.label).toBe("NSFW_SUGGESTIVE");
		expect(v?.margin).toBeCloseTo(0.63 - 0.2);
	});

	it("treats Hentai as explicit even over Sexy", () => {
		const v = nsfwVerdict(probs(0.02, 0.55, 0.15, 0.05, 0.23));
		expect(v?.label).toBe("NSFW_EXPLICIT");
	});

	it("respects a custom minMargin", () => {
		const p = probs(0.0, 0.0, 0.495, 0.505, 0.0);
		expect(nsfwVerdict(p, { minMargin: 0.005 })?.label).toBe("NSFW_EXPLICIT");
	});
});

describe("nsfwDetection", () => {
	it("emits a full-frame detection carrying the margin as score", () => {
		const v = nsfwVerdict([0, 0, 0.2, 0.6, 0.2]);
		expect(v).not.toBeNull();
		const d = nsfwDetection(v!);
		expect(d.label).toBe("NSFW_EXPLICIT");
		expect(d.box).toEqual([0, 0, 1, 1]);
		expect(d.score).toBeCloseTo(0.4);
	});
});
