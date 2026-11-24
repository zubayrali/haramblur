import { LABEL_GROUPS } from "./labels";
import type { Detection, VerdictDecision } from "./protocol";
import type { Settings } from "./settings";

/**
 * Pure classification → verdict policy (docs/PLAN.md "Policy").
 *
 * - `blurFemale` gates female-labelled classes, `blurMale` gates male ones.
 * - Exposed nudity of any person is always blurred (the gender toggles only
 *   affect faces and covered parts).
 * - `strictness` 0..1 maps to `threshold = lerp(0.6, 0.2, strictness)` for
 *   faces; exposed nudity uses a lower threshold.
 * - When the extension is off, everything is `clear`.
 *
 * The per-site on/off check lives in the content pipeline, not here.
 */

/** Minimum settings the policy needs. */
export type PolicySettings = Pick<Settings, "status" | "blurFemale" | "blurMale" | "strictness">;

export const FACE_THRESHOLD_MAX = 0.6;
export const FACE_THRESHOLD_MIN = 0.2;
export const EXPOSED_THRESHOLD_MAX = 0.4;
export const EXPOSED_THRESHOLD_MIN = 0.15;

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const clamp01 = (n: number) => Math.min(1, Math.max(0, n));

/** Confidence threshold for face/gendered classes at a given strictness. */
export function faceThreshold(strictness: number): number {
	return lerp(FACE_THRESHOLD_MAX, FACE_THRESHOLD_MIN, clamp01(strictness));
}

/** Confidence threshold for exposed-nudity classes — always lower than faces. */
export function exposedThreshold(strictness: number): number {
	return lerp(EXPOSED_THRESHOLD_MAX, EXPOSED_THRESHOLD_MIN, clamp01(strictness));
}

/**
 * Absolute minimum scores for labels whose moderate-confidence outputs are
 * noise-dominated. The genitalia classes false-positive on animals and
 * paintings (eval fixtures: a cat photo scores MALE_GENITALIA_EXPOSED@0.50,
 * a Vermeer painting @0.49) while genuine exposures either score well above
 * this bar or corroborate with other exposed classes — so the floor applies
 * regardless of strictness.
 */
const LABEL_SCORE_FLOOR: Partial<Record<Detection["label"], number>> = {
	FEMALE_GENITALIA_EXPOSED: 0.55,
	MALE_GENITALIA_EXPOSED: 0.55,
};

/**
 * The score threshold a label must meet to cause a blur under these
 * settings, or null when the label can never blur (covered parts, gendered
 * classes whose toggle is off).
 */
export function blurThresholdFor(
	label: Detection["label"],
	settings: PolicySettings,
): number | null {
	let threshold: number | null;
	if (LABEL_GROUPS.exposed.has(label)) {
		threshold = exposedThreshold(settings.strictness);
	} else if (LABEL_GROUPS.female.has(label)) {
		threshold = settings.blurFemale ? faceThreshold(settings.strictness) : null;
	} else if (LABEL_GROUPS.male.has(label)) {
		threshold = settings.blurMale ? faceThreshold(settings.strictness) : null;
	} else {
		// Covered, non-gendered parts never cause a blur on their own.
		return null;
	}
	const floor = LABEL_SCORE_FLOOR[label];
	return floor != null && threshold != null ? Math.max(threshold, floor) : threshold;
}

export function decide(
	detections: readonly Detection[],
	settings: PolicySettings,
): VerdictDecision {
	if (!settings.status) return "clear";
	for (const d of detections) {
		const threshold = blurThresholdFor(d.label, settings);
		if (threshold != null && d.score >= threshold) return "blur";
	}
	return "clear";
}

/**
 * Detections that individually meet the blur bar — the regions a region-blur
 * overlay should cover. Same per-label thresholds as `decide`, so the union
 * of these boxes is exactly what would have triggered the whole-element blur.
 */
export function blurCapableDetections(
	detections: readonly Detection[],
	settings: PolicySettings,
): Detection[] {
	if (!settings.status) return [];
	return detections.filter((d) => {
		const threshold = blurThresholdFor(d.label, settings);
		return threshold != null && d.score >= threshold;
	});
}
