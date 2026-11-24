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

/**
 * Whole-image classifier labels (nsfwjs MobileNetV2) — emitted as
 * full-frame detections `[0,0,1,1]` when the suggestive/anime signal fires.
 * Not NudeNet classes; they exist so `decide()`/region blur treat them
 * uniformly with localized detections.
 */
export const NSFW_LABELS = ["NSFW_EXPLICIT", "NSFW_SUGGESTIVE"] as const;
export type NsfwLabel = (typeof NSFW_LABELS)[number];

/**
 * Engine-synthesized labels that are not NudeNet classes. `FACE_PERSON` is a
 * neutral face placeholder: emitted when a detector is confident a face is
 * present but the gender classifier abstains. It suppresses NudeNet's noisy
 * `FACE_FEMALE`/`FACE_MALE` on the same box (via the IoU filter) without
 * asserting a gender — so it never blurs on its own.
 */
export const SYNTHETIC_LABELS = ["FACE_PERSON"] as const;
export type SyntheticLabel = (typeof SYNTHETIC_LABELS)[number];

/** Any label a Detection can carry. */
export type DetectionLabel = NudeNetLabel | NsfwLabel | SyntheticLabel;

const set = (labels: readonly string[]): ReadonlySet<string> => new Set(labels);

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

/**
 * All face-kind labels (gendered + neutral). Used by the video sampler's
 * hysteresis: a blur verdict carried only by face labels is not
 * "high-confidence" — gender flip-flops need the second agreeing frame.
 */
export const FACE_KIND_LABELS: ReadonlySet<string> = set([
	"FACE_FEMALE",
	"FACE_MALE",
	"FACE_PERSON",
]);
