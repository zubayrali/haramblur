import { type DetectionLabel, LABEL_GROUPS, NSFW_LABELS } from "./labels";
import type { Detection, VerdictDecision } from "./protocol";
import type { Settings } from "./settings";

/**
 * Pure classification → verdict policy (docs/PLAN.md "Policy").
 *
 * - `blurFemale` gates female-labelled classes, `blurMale` gates male ones.
 * - `CORE_EXPOSED_LABELS` (explicit nudity) always blur regardless of the
 *   gender toggles; genitalia classes carry an absolute score floor because
 *   NudeNet false-positives on them at moderate confidence.
 * - `MODESTY_LABELS` (bare belly/feet/armpits, covered chest, male breast)
 *   only blur when `strictModesty` is on — NudeNet precision on them is poor.
 * - `strictness` 0..1 maps to `threshold = lerp(0.6, 0.2, strictness)` for
 *   faces; exposed nudity uses a lower threshold.
 * - When the extension is off, everything is `clear`.
 *
 * The per-site on/off check lives in the content pipeline, not here.
 */

/** Minimum settings the policy needs. */
export type PolicySettings = Pick<
	Settings,
	"status" | "blurFemale" | "blurMale" | "strictness" | "strictModesty" | "skipChildren"
>;

export const FACE_THRESHOLD_MAX = 0.6;
export const FACE_THRESHOLD_MIN = 0.2;
export const EXPOSED_THRESHOLD_MAX = 0.4;
export const EXPOSED_THRESHOLD_MIN = 0.15;
export const NSFW_THRESHOLD_MAX = 0.4;
export const NSFW_THRESHOLD_MIN = 0.15;

/**
 * Reference precision floor for the noisiest classes — the strictness-0.5
 * operating point of the genitalia/modesty curves stays at or above it so
 * default behavior does not regress (dev-split negatives cluster at 0.36–0.52).
 */
export const LABEL_SCORE_FLOOR = 0.55;

/**
 * Male-context suppression: a confident `FACE_MALE` makes a low-confidence
 * `FEMALE_BREAST_EXPOSED` directly below his chin anatomically implausible —
 * it's his own lapel/shirt/skin, not a woman. Only the torso zone counts
 * (a woman beside him keeps her own geometry), and only below
 * `MALE_CONTEXT_MAX_TARGET_SCORE` — strong detections always stand.
 */
const MALE_CONTEXT_MIN_SCORE = 0.6;
const MALE_CONTEXT_MAX_TARGET_SCORE = 0.75;
const MALE_TORSO_LABELS: ReadonlySet<string> = new Set(["FEMALE_BREAST_EXPOSED"]);

function suppressedByMaleContext(d: Detection, all: readonly Detection[]): boolean {
	if (!MALE_TORSO_LABELS.has(d.label) || d.score >= MALE_CONTEXT_MAX_TARGET_SCORE) {
		return false;
	}
	const [dx, dy, dw] = d.box;
	const cx = dx + dw / 2;
	for (const m of all) {
		if (m.label !== "FACE_MALE" || m.score < MALE_CONTEXT_MIN_SCORE) continue;
		const [mx, my, mw, mh] = m.box;
		const inX = cx >= mx - mw * 0.35 && cx <= mx + mw * 1.35;
		const torsoY = dy >= my + mh * 0.6 && dy <= my + mh * 2.6;
		if (inX && torsoY) return true;
	}
	return false;
}

/** Detections that count toward a verdict — context-suppressed ones don't. */
function effectiveDetections(detections: readonly Detection[]): readonly Detection[] {
	return detections.filter((d) => !suppressedByMaleContext(d, detections));
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const clamp01 = (n: number) => Math.min(1, Math.max(0, n));

/**
 * Per-class strictness curve (DP4). `threshold = max − (max−min)·strictness^pow`.
 * `pow > 1` keeps thresholds high through the mid-range and drops them steeply
 * near strictness 1 — the shape the dev-split score distributions demand:
 * genitalia/breast false positives cluster at 0.36–0.52, so mid-strictness
 * must protect precision, while real positives exist as low as 0.17 and only
 * a high-strictness floor near upstream's ~0.1 can recall them.
 */
export interface ThresholdCurve {
	/** Threshold at strictness 0 (precision end). */
	max: number;
	/** Threshold at strictness 1 (sensitivity end, upstream parity ~0.1). */
	min: number;
	/** Curve exponent; 1 = linear, 2 = steep drop at the sensitive end. */
	pow: number;
}

export function curveThreshold(curve: ThresholdCurve, strictness: number): number {
	return curve.max - (curve.max - curve.min) * clamp01(strictness) ** curve.pow;
}

const FACE_CURVE: ThresholdCurve = { max: 0.6, min: 0.2, pow: 1 };
const NSFW_CURVE: ThresholdCurve = { max: 0.4, min: 0.15, pow: 1 };
/** Genitalia: dev positives at 0.17/0.45 (F) and 0.70 (M); negatives cluster
 * at 0.36–0.52. pow 3 keeps the default (0.5) floor at 0.58 ≈ the old hard
 * floor's effect (0.55), while strictness 1 reaches 0.12 and recalls the 0.17 positive. */
const GENITALIA_CURVE: ThresholdCurve = { max: 0.65, min: 0.12, pow: 3 };
/** Breast/buttocks/anus: skin-texture negatives up to ~0.44, but a real breast
 * positive at 0.29. Replaces the old flat `+0.2` margin — nearly identical at
 * default strictness (0.45 vs 0.475), much more sensitive at the strict end. */
const NOISY_EXPOSED_CURVE: ThresholdCurve = { max: 0.6, min: 0.15, pow: 2 };
/** Modesty classes are the noisiest (negatives to 0.81) — even the sensitive
 * end stays conservative, and `strictModesty` still gates them entirely. */
const MODESTY_CURVE: ThresholdCurve = { max: 0.6, min: 0.35, pow: 1.5 };

/** Per-label curve table — the single source of truth for strictness scaling. */
export const THRESHOLD_CURVES: Readonly<Partial<Record<DetectionLabel, ThresholdCurve>>> = {
	FEMALE_GENITALIA_EXPOSED: GENITALIA_CURVE,
	MALE_GENITALIA_EXPOSED: GENITALIA_CURVE,
	FEMALE_BREAST_EXPOSED: NOISY_EXPOSED_CURVE,
	BUTTOCKS_EXPOSED: NOISY_EXPOSED_CURVE,
	ANUS_EXPOSED: NOISY_EXPOSED_CURVE,
	FEET_EXPOSED: MODESTY_CURVE,
	BELLY_EXPOSED: MODESTY_CURVE,
	ARMPITS_EXPOSED: MODESTY_CURVE,
	MALE_BREAST_EXPOSED: MODESTY_CURVE,
	NSFW_EXPLICIT: NSFW_CURVE,
	NSFW_SUGGESTIVE: NSFW_CURVE,
};

/** Faces estimated younger than this are skipped when `skipChildren` is on. */
export const CHILD_AGE_MAX = 15;

/** A face detection the gender model believes is a child — never blurs alone. */
function isChildFace(d: Detection, settings: PolicySettings): boolean {
	return (
		settings.skipChildren &&
		(d.label === "FACE_FEMALE" || d.label === "FACE_MALE") &&
		d.age != null &&
		d.age < CHILD_AGE_MAX
	);
}

/** Confidence threshold for face/gendered classes at a given strictness. */
export function faceThreshold(strictness: number): number {
	return lerp(FACE_THRESHOLD_MAX, FACE_THRESHOLD_MIN, clamp01(strictness));
}

/** Confidence threshold for exposed-nudity classes — always lower than faces. */
export function exposedThreshold(strictness: number): number {
	return lerp(EXPOSED_THRESHOLD_MAX, EXPOSED_THRESHOLD_MIN, clamp01(strictness));
}

const NSFW_LABEL_SET: ReadonlySet<string> = new Set(NSFW_LABELS);

/**
 * Whole-image NSFW detections carry a *margin* score (nsfw prob − sfw prob),
 * not a calibrated probability — hence their own threshold band. Always on:
 * the classifier only emits when it already beat SFW.
 */
export function nsfwThreshold(strictness: number): number {
	return lerp(NSFW_THRESHOLD_MAX, NSFW_THRESHOLD_MIN, clamp01(strictness));
}

/**
 * Confidence threshold at which `label` causes a blur under `settings`, or
 * `null` when the label can never blur (covered non-gendered parts, or
 * modesty classes while `strictModesty` is off). This is the single source of
 * truth shared by `decide()` and the region-blur layer.
 */
export function blurThresholdFor(label: DetectionLabel, settings: PolicySettings): number | null {
	if (NSFW_LABEL_SET.has(label)) return nsfwThreshold(settings.strictness);
	if (LABEL_GROUPS.coreExposed.has(label)) {
		const curve = THRESHOLD_CURVES[label];
		return curve != null
			? curveThreshold(curve, settings.strictness)
			: exposedThreshold(settings.strictness);
	}
	if (LABEL_GROUPS.modesty.has(label)) {
		if (!settings.strictModesty) return null;
		const curve = THRESHOLD_CURVES[label];
		return curve != null
			? curveThreshold(curve, settings.strictness)
			: Math.max(faceThreshold(settings.strictness), LABEL_SCORE_FLOOR);
	}
	if (LABEL_GROUPS.female.has(label)) {
		return settings.blurFemale ? faceThreshold(settings.strictness) : null;
	}
	if (LABEL_GROUPS.male.has(label)) {
		return settings.blurMale ? faceThreshold(settings.strictness) : null;
	}
	// Covered, non-gendered parts never cause a blur on their own.
	return null;
}

/** Detections that would contribute to a blur verdict under `settings`. */
export function blurCapableDetections(
	detections: readonly Detection[],
	settings: PolicySettings,
): Detection[] {
	return effectiveDetections(detections).filter((d) => {
		if (isChildFace(d, settings)) return false;
		const thr = blurThresholdFor(d.label, settings);
		return thr != null && d.score >= thr;
	});
}

export function decide(
	detections: readonly Detection[],
	settings: PolicySettings,
): VerdictDecision {
	if (!settings.status) return "clear";
	for (const d of effectiveDetections(detections)) {
		if (isChildFace(d, settings)) continue;
		const thr = blurThresholdFor(d.label, settings);
		if (thr != null && d.score >= thr) return "blur";
	}
	return "clear";
}
