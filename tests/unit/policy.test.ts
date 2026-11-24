import { describe, expect, it } from "vitest";
import { decide, exposedThreshold, faceThreshold } from "../../src/shared/policy";
import type { Detection } from "../../src/shared/protocol";
import { DEFAULT_SETTINGS } from "../../src/shared/settings";

const base = {
	status: true,
	blurFemale: true,
	blurMale: false,
	strictness: 0.5,
};

const det = (label: Detection["label"], score: number): Detection => ({
	label,
	score,
	box: [0, 0, 1, 1],
});

describe("decide", () => {
	it("clears when there are no detections", () => {
		expect(decide([], base)).toBe("clear");
	});

	it("clears when the extension is off, even for exposed nudity", () => {
		const dets = [det("ANUS_EXPOSED", 0.99)];
		expect(decide(dets, { ...base, status: false })).toBe("clear");
	});

	it("blurs a female face when blurFemale is on", () => {
		expect(decide([det("FACE_FEMALE", 0.7)], base)).toBe("blur");
	});

	it("clears a female face when blurFemale is off", () => {
		const dets = [det("FACE_FEMALE", 0.7)];
		expect(decide(dets, { ...base, blurFemale: false })).toBe("clear");
	});

	it("blurs a male face only when blurMale is on", () => {
		const dets = [det("FACE_MALE", 0.7)];
		expect(decide(dets, base)).toBe("clear");
		expect(decide(dets, { ...base, blurMale: true })).toBe("blur");
	});

	it("always blurs exposed nudity regardless of gender toggles", () => {
		const dets = [det("FEMALE_BREAST_EXPOSED", 0.5)];
		expect(decide(dets, { ...base, blurFemale: false, blurMale: false })).toBe("blur");
		expect(decide([det("MALE_GENITALIA_EXPOSED", 0.6)], base)).toBe("blur");
	});

	it("genitalia classes need a higher absolute score (eval false-positive floor)", () => {
		// Cat photo → MALE_GENITALIA_EXPOSED@0.50, Vermeer → @0.49: below the
		// 0.55 floor these must not blur even though they clear the exposed bar.
		expect(decide([det("MALE_GENITALIA_EXPOSED", 0.5)], base)).toBe("clear");
		expect(decide([det("FEMALE_GENITALIA_EXPOSED", 0.5)], base)).toBe("clear");
		expect(decide([det("MALE_GENITALIA_EXPOSED", 0.55)], base)).toBe("blur");
		// The floor is absolute — it holds at maximum strictness too.
		expect(decide([det("MALE_GENITALIA_EXPOSED", 0.5)], { ...base, strictness: 1 })).toBe("clear");
	});

	it("never blurs covered, non-gendered labels on their own", () => {
		for (const label of [
			"BELLY_COVERED",
			"FEET_COVERED",
			"ARMPITS_COVERED",
			"ANUS_COVERED",
			"BUTTOCKS_COVERED",
		] as const) {
			expect(decide([det(label, 0.99)], base), label).toBe("clear");
		}
	});

	it("blurs gendered covered labels only when that gender's toggle is on", () => {
		const dets = [det("FEMALE_GENITALIA_COVERED", 0.7)];
		expect(decide(dets, base)).toBe("blur");
		expect(decide(dets, { ...base, blurFemale: false })).toBe("clear");
	});

	it("uses a lower threshold for exposed nudity than for faces", () => {
		// Between the two thresholds at strictness 0: exposed blurs, faces don't.
		const s = { ...base, strictness: 0 };
		expect(faceThreshold(0)).toBeCloseTo(0.6);
		expect(exposedThreshold(0)).toBeCloseTo(0.4);
		expect(decide([det("FACE_FEMALE", 0.5)], s)).toBe("clear");
		expect(decide([det("ANUS_EXPOSED", 0.45)], s)).toBe("blur");
	});

	it("higher strictness lowers face thresholds via lerp(0.6, 0.2, s)", () => {
		const lenient = { ...base, strictness: 0 };
		const strict = { ...base, strictness: 1 };
		expect(decide([det("FACE_FEMALE", 0.3)], lenient)).toBe("clear");
		expect(decide([det("FACE_FEMALE", 0.3)], strict)).toBe("blur");
	});

	it("applies the same per-class gendered threshold to covered female parts", () => {
		const s = { ...base, strictness: 1 };
		expect(decide([det("FEMALE_BREAST_COVERED", 0.25)], s)).toBe("blur");
		expect(decide([det("FEMALE_BREAST_COVERED", 0.15)], s)).toBe("clear");
	});

	it("clamps strictness outside 0..1", () => {
		expect(faceThreshold(-1)).toBeCloseTo(0.6);
		expect(faceThreshold(2)).toBeCloseTo(0.2);
	});
});

describe("defaults", () => {
	it("DEFAULT_SETTINGS satisfies PolicySettings", () => {
		expect(decide([det("FACE_FEMALE", 0.9)], DEFAULT_SETTINGS)).toBe("blur");
		expect(decide([det("FACE_MALE", 0.9)], DEFAULT_SETTINGS)).toBe("clear");
	});
});
