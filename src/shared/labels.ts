/**
 * NudeNet `320n.onnx` class labels in model order (see docs/ENGINE_NOTES.md).
 * The index in this array is the class id emitted by the model.
 */
export const NUDENET_LABELS = [
	"FEMALE_GENITALIA_COVERED",
	"FACE_FEMALE",
	"BUTTOCKS_EXPOSED",
	"FEMALE_BREAST_EXPOSED",
	"FEMALE_GENITALIA_EXPOSED",
	"MALE_BREAST_EXPOSED",
	"ANUS_EXPOSED",
	"FEET_EXPOSED",
	"BELLY_COVERED",
	"FEET_COVERED",
	"ARMPITS_COVERED",
	"ARMPITS_EXPOSED",
	"FACE_MALE",
	"BELLY_EXPOSED",
	"MALE_GENITALIA_EXPOSED",
	"ANUS_COVERED",
	"FEMALE_BREAST_COVERED",
	"BUTTOCKS_COVERED",
] as const;

export type NudeNetLabel = (typeof NUDENET_LABELS)[number];

const set = (labels: readonly NudeNetLabel[]): ReadonlySet<NudeNetLabel> => new Set(labels);

/** Labels that describe a woman (face or body parts, covered or exposed). */
export const FEMALE_LABELS = set([
	"FACE_FEMALE",
	"FEMALE_BREAST_EXPOSED",
	"FEMALE_GENITALIA_COVERED",
	"FEMALE_GENITALIA_EXPOSED",
] as const);

/** Labels that describe a man. */
export const MALE_LABELS = set([
	"FACE_MALE",
	"MALE_BREAST_EXPOSED",
	"MALE_GENITALIA_EXPOSED",
] as const);

/**
 * Explicit nudity — always blurred regardless of gender toggles.
 * Genitalia/anus classes carry an absolute score floor in policy.ts because
 * NudeNet false-positives on them at moderate confidence (animals, art).
 */
export const CORE_EXPOSED_LABELS = set([
	"FEMALE_BREAST_EXPOSED",
	"FEMALE_GENITALIA_EXPOSED",
	"MALE_GENITALIA_EXPOSED",
	"BUTTOCKS_EXPOSED",
	"ANUS_EXPOSED",
] as const);

/**
 * Secondary exposure / modesty classes (bare belly, armpits, feet, covered
 * chest, male breast). Eval showed ~0 precision on these at default
 * thresholds — they only blur when the user opts into `strictModesty`.
 */
export const MODESTY_LABELS = set([
	"BELLY_EXPOSED",
	"MALE_BREAST_EXPOSED",
	"ARMPITS_EXPOSED",
	"FEET_EXPOSED",
	"FEMALE_BREAST_COVERED",
] as const);

/** Covered body parts; never blurred on their own unless gendered (see policy). */
export const COVERED_LABELS = set([
	"FEMALE_GENITALIA_COVERED",
	"BELLY_COVERED",
	"FEET_COVERED",
	"ARMPITS_COVERED",
	"ANUS_COVERED",
	"FEMALE_BREAST_COVERED",
	"BUTTOCKS_COVERED",
] as const);

export const LABEL_GROUPS = {
	female: FEMALE_LABELS,
	male: MALE_LABELS,
	coreExposed: CORE_EXPOSED_LABELS,
	modesty: MODESTY_LABELS,
	covered: COVERED_LABELS,
} as const;
