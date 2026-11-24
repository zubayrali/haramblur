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
	"FEMALE_BREAST_COVERED",
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

/** Any exposed body part, regardless of who it belongs to. */
export const EXPOSED_LABELS = set([
	"BUTTOCKS_EXPOSED",
	"FEMALE_BREAST_EXPOSED",
	"FEMALE_GENITALIA_EXPOSED",
	"MALE_BREAST_EXPOSED",
	"ANUS_EXPOSED",
	"FEET_EXPOSED",
	"ARMPITS_EXPOSED",
	"BELLY_EXPOSED",
	"MALE_GENITALIA_EXPOSED",
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
	exposed: EXPOSED_LABELS,
	covered: COVERED_LABELS,
} as const;
