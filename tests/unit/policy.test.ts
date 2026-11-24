import { describe, expect, it } from "vitest";
import {
	blurCapableDetections,
	blurThresholdFor,
	decide,
	exposedThreshold,
	faceThreshold,
	LABEL_SCORE_FLOOR,
} from "../../src/shared/policy";
import type { Detection } from "../../src/shared/protocol";
import { DEFAULT_SETTINGS } from "../../src/shared/settings";

const base = {
	status: true,
	blurFemale: true,
	blurMale: false,
	strictness: 0.5,
	strictModesty: false,
	skipChildren: true,
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

	it("always blurs core exposed nudity regardless of gender toggles", () => {
		const dets = [det("FEMALE_BREAST_EXPOSED", 0.5)];
		expect(decide(dets, { ...base, blurFemale: false, blurMale: false })).toBe("blur");
		expect(decide([det("BUTTOCKS_EXPOSED", 0.5)], base)).toBe("blur");
	});

	it("applies an absolute floor to genitalia classes", () => {
		// Cat-photo FP territory: below the floor → clear, above → blur.
		expect(decide([det("MALE_GENITALIA_EXPOSED", LABEL_SCORE_FLOOR - 0.01)], base)).toBe("clear");
		expect(decide([det("MALE_GENITALIA_EXPOSED", LABEL_SCORE_FLOOR + 0.01)], base)).toBe("blur");
		expect(decide([det("FEMALE_GENITALIA_EXPOSED", LABEL_SCORE_FLOOR - 0.01)], base)).toBe("clear");
		// Non-genitalia core classes have no floor.
		expect(decide([det("ANUS_EXPOSED", 0.3)], base)).toBe("blur");
	});

	it("ignores modesty labels unless strictModesty is on", () => {
		for (const label of [
			"BELLY_EXPOSED",
			"MALE_BREAST_EXPOSED",
			"ARMPITS_EXPOSED",
			"FEET_EXPOSED",
			"FEMALE_BREAST_COVERED",
		] as const) {
			expect(decide([det(label, 0.99)], base), label).toBe("clear");
			expect(decide([det(label, 0.99)], { ...base, strictModesty: true }), label).toBe("blur");
		}
	});

	it("floors modesty labels even when strictModesty is on", () => {
		const s = { ...base, strictModesty: true };
		expect(decide([det("FEET_EXPOSED", LABEL_SCORE_FLOOR - 0.01)], s)).toBe("clear");
		expect(decide([det("FEET_EXPOSED", LABEL_SCORE_FLOOR + 0.01)], s)).toBe("blur");
	});

	it("never blurs covered, non-gendered labels on their own", () => {
		for (const label of [
			"BELLY_COVERED",
			"FEET_COVERED",
			"ARMPITS_COVERED",
			"ANUS_COVERED",
			"BUTTOCKS_COVERED",
		] as const) {
			expect(decide([det(label, 0.99)], { ...base, strictModesty: true }), label).toBe("clear");
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

	it("skipChildren suppresses child face detections only", () => {
		const child = { ...det("FACE_FEMALE", 0.9), age: 10 };
		const adult = { ...det("FACE_FEMALE", 0.9), age: 30 };
		expect(decide([child], base)).toBe("clear");
		expect(decide([child], { ...base, skipChildren: false })).toBe("blur");
		expect(decide([adult], base)).toBe("blur");
		// A child face never suppresses explicit nudity on its own detection,
		// and non-face labels ignore age entirely.
		expect(decide([{ ...det("FEMALE_BREAST_EXPOSED", 0.6), age: 10 }], base)).toBe("blur");
		expect(decide([{ ...det("FACE_MALE", 0.9), age: 10 }], { ...base, blurMale: true })).toBe(
			"clear",
		);
	});

	it("clamps strictness outside 0..1", () => {
		expect(faceThreshold(-1)).toBeCloseTo(0.6);
		expect(faceThreshold(2)).toBeCloseTo(0.2);
	});
});

describe("blurThresholdFor / blurCapableDetections", () => {
	it("returns null for labels that can never blur", () => {
		expect(blurThresholdFor("BELLY_COVERED", base)).toBeNull();
		expect(blurThresholdFor("FEET_EXPOSED", base)).toBeNull(); // modesty off
		expect(blurThresholdFor("FACE_FEMALE", { ...base, blurFemale: false })).toBeNull();
	});

	it("filters detections to exactly those decide() would blur on", () => {
		const dets = [
			det("FACE_FEMALE", 0.9),
			det("FACE_FEMALE", 0.1),
			det("FEET_EXPOSED", 0.99),
			det("FEMALE_GENITALIA_EXPOSED", 0.6),
			det("BELLY_COVERED", 0.99),
		];
		const capable = blurCapableDetections(dets, base);
		expect(capable.map((d) => d.score)).toEqual([0.9, 0.6]);
	});
});

describe("defaults", () => {
	it("DEFAULT_SETTINGS satisfies PolicySettings", () => {
		expect(decide([det("FACE_FEMALE", 0.9)], DEFAULT_SETTINGS)).toBe("blur");
		expect(decide([det("FACE_MALE", 0.9)], DEFAULT_SETTINGS)).toBe("clear");
	});
});
