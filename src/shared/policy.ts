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
 * Absolute score floor for the genitalia classes. NudeNet fires
 * `*_GENITALIA_EXPOSED` on animals and artwork at ~0.5; real positives score
 * higher. Also applied to modesty labels, which are noisy by nature.
 */
export const LABEL_SCORE_FLOOR = 0.55;
const FLOOR_LABELS: ReadonlySet<string> = new Set([
	"FEMALE_GENITALIA_EXPOSED",
	"MALE_GENITALIA_EXPOSED",
]);

/**
 * Breast/buttocks/anus classes fire on skin-tone textures — folded lapels,
 * hands, chins — at ~0.3–0.45 (observed live on YouTube thumbnails). They
 * require a margin above the exposed threshold rather than a flat floor, so
 * `strictness` still scales them at both ends.
 */
const NOISY_EXPOSED_MARGIN = 0.2;
const NOISY_EXPOSED_LABELS: ReadonlySet<string> = new Set([
	"FEMALE_BREAST_EXPOSED",
	"BUTTOCKS_EXPOSED",
	"ANUS_EXPOSED",
]);

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
		if (FLOOR_LABELS.has(label)) {
			return Math.max(exposedThreshold(settings.strictness), LABEL_SCORE_FLOOR);
		}
		if (NOISY_EXPOSED_LABELS.has(label)) {
			return Math.min(1, exposedThreshold(settings.strictness) + NOISY_EXPOSED_MARGIN);
		}
		return exposedThreshold(settings.strictness);
	}
	if (LABEL_GROUPS.modesty.has(label)) {
		if (!settings.strictModesty) return null;
		return Math.max(faceThreshold(settings.strictness), LABEL_SCORE_FLOOR);
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
