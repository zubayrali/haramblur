import { describe, expect, it } from "vitest";
import {
	blurCapableDetections,
	blurThresholdFor,
	decide,
	exposedThreshold,
	faceThreshold,
	LABEL_SCORE_FLOOR,
	nsfwThreshold,
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

	it("keeps genitalia precision at default strictness and opens up at strictness 1", () => {
		// Default (0.5): curve floor ≈0.58 — cat-photo FP territory stays clear.
		expect(decide([det("MALE_GENITALIA_EXPOSED", 0.52)], base)).toBe("clear");
		expect(decide([det("MALE_GENITALIA_EXPOSED", 0.7)], base)).toBe("blur");
		// Strictness 1: floor drops toward upstream's ~0.1 — the dev-split 0.17
		// positive is recalled; mid-confidence FPs are the accepted tradeoff.
		const strict = { ...base, strictness: 1 };
		expect(blurThresholdFor("FEMALE_GENITALIA_EXPOSED", strict)).toBeCloseTo(0.12);
		expect(decide([det("FEMALE_GENITALIA_EXPOSED", 0.17)], strict)).toBe("blur");
	});

	it("requires a noise margin for breast/buttocks/anus classes", () => {
		// At strictness 0.5 the exposed threshold is 0.275; noisy classes need
		// +0.2 → 0.475. The 0.3–0.45 skin-texture FPs stay clear.
		expect(decide([det("FEMALE_BREAST_EXPOSED", 0.3)], base)).toBe("clear");
		expect(decide([det("BUTTOCKS_EXPOSED", 0.45)], base)).toBe("clear");
		expect(decide([det("ANUS_EXPOSED", 0.45)], base)).toBe("clear");
		expect(decide([det("FEMALE_BREAST_EXPOSED", 0.5)], base)).toBe("blur");
		// The margin still scales with strictness (lenient end loosens it).
		expect(decide([det("BUTTOCKS_EXPOSED", 0.45)], { ...base, strictness: 1 })).toBe("blur");
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

	it("scales modesty labels with strictness when strictModesty is on", () => {
		const s = { ...base, strictModesty: true };
		// Default strictness: curve sits at ≈0.51 — low-confidence noise stays clear.
		expect(decide([det("FEET_EXPOSED", 0.44)], s)).toBe("clear");
		expect(decide([det("FEET_EXPOSED", 0.6)], s)).toBe("blur");
		expect(decide([det("FEET_EXPOSED", 0.44)], { ...s, strictness: 1 })).toBe("blur");
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

	it("noisy exposed classes match the face bar at strictness 0 and drop below it at strictness 1", () => {
		const lenient = { ...base, strictness: 0 };
		expect(faceThreshold(0)).toBeCloseTo(0.6);
		expect(blurThresholdFor("BUTTOCKS_EXPOSED", lenient)).toBeCloseTo(0.6);
		expect(decide([det("BUTTOCKS_EXPOSED", 0.55)], lenient)).toBe("clear");
		expect(decide([det("BUTTOCKS_EXPOSED", 0.65)], lenient)).toBe("blur");
		// Strict end: 0.15 — the dev-split 0.29 breast positive is recalled.
		const strict = { ...base, strictness: 1 };
		expect(blurThresholdFor("FEMALE_BREAST_EXPOSED", strict)).toBeCloseTo(0.15);
		expect(decide([det("FEMALE_BREAST_EXPOSED", 0.29)], strict)).toBe("blur");
		// Default strictness stays at ≈0.49 so the 0.3–0.45 skin-texture FPs stay clear.
		expect(decide([det("FEMALE_BREAST_EXPOSED", 0.45)], base)).toBe("clear");
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

	it("suppresses low-confidence breast hits in a male face's torso zone", () => {
		// The YouTube-lapel FP: confident FACE_MALE with a sub-0.75
		// FEMALE_BREAST_EXPOSED centered under his chin.
		const maleFace: Detection = {
			label: "FACE_MALE",
			score: 0.9,
			box: [0.4, 0.1, 0.2, 0.25],
		};
		const lapel: Detection = {
			label: "FEMALE_BREAST_EXPOSED",
			score: 0.5, // above the 0.475 noise margin, below the 0.75 suppression ceiling
			box: [0.42, 0.32, 0.16, 0.2], // top ≈ chin level, centered under face
		};
		expect(decide([maleFace, lapel], base)).toBe("clear");
		expect(blurCapableDetections([maleFace, lapel], base)).toHaveLength(0);
		// Off to the side (a woman beside him) → not suppressed.
		const besideHim: Detection = { ...lapel, box: [0.05, 0.32, 0.16, 0.2] };
		expect(decide([maleFace, besideHim], base)).toBe("blur");
		// High-confidence detections always stand, even in the torso zone.
		const strong: Detection = { ...lapel, score: 0.8 };
		expect(decide([maleFace, strong], base)).toBe("blur");
		// A weak male face can't suppress anything.
		const weakMale: Detection = { ...maleFace, score: 0.4 };
		expect(decide([weakMale, lapel], base)).toBe("blur");
	});

	it("FACE_PERSON never blurs but suppresses nothing else", () => {
		expect(decide([det("FACE_PERSON", 0.99)], base)).toBe("clear");
		expect(blurThresholdFor("FACE_PERSON", base)).toBeNull();
		// Mixed with a real trigger it changes nothing.
		expect(decide([det("FACE_PERSON", 0.9), det("FACE_FEMALE", 0.9)], base)).toBe("blur");
	});

	it("NSFW labels blur on their margin score, strictness-scaled", () => {
		// At strictness 0.5: nsfwThreshold = lerp(0.4, 0.15, 0.5) = 0.275.
		expect(nsfwThreshold(0.5)).toBeCloseTo(0.275);
		expect(decide([det("NSFW_EXPLICIT", 0.3)], base)).toBe("blur");
		expect(decide([det("NSFW_SUGGESTIVE", 0.2)], base)).toBe("clear");
		// Lenient strictness needs a bigger margin; strict loosens it.
		expect(decide([det("NSFW_EXPLICIT", 0.3)], { ...base, strictness: 0 })).toBe("clear");
		expect(decide([det("NSFW_EXPLICIT", 0.2)], { ...base, strictness: 1 })).toBe("blur");
		// NSFW labels ignore the gender toggles — explicit content is never opt-out.
		expect(
			decide([det("NSFW_EXPLICIT", 0.3)], { ...base, blurFemale: false, blurMale: false }),
		).toBe("blur");
		expect(blurThresholdFor("NSFW_SUGGESTIVE", base)).toBeCloseTo(0.275);
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
