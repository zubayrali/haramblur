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

export function decide(
	detections: readonly Detection[],
	settings: PolicySettings,
): VerdictDecision {
	if (!settings.status) return "clear";
	const face = faceThreshold(settings.strictness);
	const exposed = exposedThreshold(settings.strictness);
	for (const d of detections) {
		if (LABEL_GROUPS.exposed.has(d.label)) {
			if (d.score >= exposed) return "blur";
			continue;
		}
		if (LABEL_GROUPS.female.has(d.label)) {
			if (settings.blurFemale && d.score >= face) return "blur";
			continue;
		}
		if (LABEL_GROUPS.male.has(d.label)) {
			if (settings.blurMale && d.score >= face) return "blur";
		}
		// Covered, non-gendered parts never cause a blur on their own.
	}
	return "clear";
}
