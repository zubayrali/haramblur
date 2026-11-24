/**
 * BlazeFace short-range face *detector* (`blaze-face.onnx`, Apache-2.0).
 *
 * NudeNet's `FACE_FEMALE`/`FACE_MALE` labels conflate "is there a face" with
 * "whose face" in one weak classifier — eval shows missed faces on
 * elderly/group photos and phantom gender labels on portraits. This module
 * runs a dedicated detector (boxes only, no gender) and leaves gender to
 * FastFace, mirroring the original extension's blazeface+faceres split.
 *
 * Tensor contract (verified against the export):
 * - Input `image` `[1, 3, 128, 128]` float32, RGB, 0..1 (NOT normalized),
 *   NCHW — the bitmap is stretched to square, matching how the graph was
 *   exported (decode + NMS + thresholding are fused inside).
 * - Scalar inputs `conf_threshold`, `iou_threshold`, `max_detections`.
 * - Output `selectedBoxes` `[1, N, 16]` (shape collapses to `[1, 16]` for a
 *   single hit and `[1, 0, 16]` for none). Per row: `top_y, top_x, bot_y,
 *   bot_x` normalized 0..1 followed by 6 landmark pairs (unused here).
 *
 * The graph emits no per-detection score, so confidence is recovered by
 * running the same tensor at several `conf_threshold`s: a box seen at a
 * higher threshold is a more confident face. `FACE_DET_PASSES` maps each
 * pass to the `Detection.score` it produces, which keeps the strictness
 * slider meaningful for faces.
 */
import type { Detection } from "../shared/protocol";

export const FACE_DET_SIZE = 128;
export const FACE_DET_IOU = 0.3;
export const FACE_DET_MAX = 100;

/**
 * Confidence passes, lowest → highest threshold, each mapped to the score a
 * detection earns. Chosen from fixture probes: real faces survive ≥0.4 while
 * cat/painting false positives live below ~0.35.
 */
export const FACE_DET_PASSES = [
	{ conf: 0.4, score: 0.45 },
	{ conf: 0.55, score: 0.65 },
	{ conf: 0.7, score: 0.85 },
] as const;

/** Boxes from different passes are the same face above this IoU. */
const FACE_MATCH_IOU = 0.5;
/** A NudeNet FACE_* box is suppressed when it overlaps a detector box at
 * least this much — the dedicated pipeline's verdict wins. */
const FACE_SUPPRESS_IOU = 0.2;
/**
 * Minimum NudeNet `FACE_*` score that corroborates a detector box. Both
 * models false-positive on animal faces (dog fixture: NudeNet 0.23 +
 * BlazeFace ≥0.8 on the same snout), so the corroboration bar sits above
 * weak animal-face suspicion while real portrait faces score 0.4+.
 */
export const FACE_CORROBORATE_MIN = 0.3;

export type NormBox = Detection["box"];

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

export function boxIou(a: NormBox, b: NormBox): number {
	const ax1 = a[0] + a[2];
	const ay1 = a[1] + a[3];
	const bx1 = b[0] + b[2];
	const by1 = b[1] + b[3];
	const iw = Math.max(0, Math.min(ax1, bx1) - Math.max(a[0], b[0]));
	const ih = Math.max(0, Math.min(ay1, by1) - Math.max(a[1], b[1]));
	const inter = iw * ih;
	if (inter <= 0) return 0;
	const union = a[2] * a[3] + b[2] * b[3] - inter;
	return union > 0 ? inter / union : 0;
}

/**
 * Same-face test for cross-source comparison: IoU overlap, or the smaller
 * box's center sitting inside the larger one (detectors disagree on box
 * tightness — a loose NudeNet box around a tight BlazeFace box has low IoU
 * but the same center).
 */
export function sameFace(a: NormBox, b: NormBox): boolean {
	if (boxIou(a, b) > FACE_SUPPRESS_IOU) return true;
	const small = a[2] * a[3] <= b[2] * b[3] ? a : b;
	const large = small === a ? b : a;
	const cx = small[0] + small[2] / 2;
	const cy = small[1] + small[3] / 2;
	return cx >= large[0] && cx <= large[0] + large[2] && cy >= large[1] && cy <= large[1] + large[3];
}

/**
 * Parse the fused `selectedBoxes` output into normalized `[x, y, w, h]`
 * boxes. Handles the shape variants the export produces (`[1,N,16]`,
 * `[1,16]`, `[1,0,16]`).
 */
export function parseFaceBoxes(data: ArrayLike<number>, dims: readonly number[]): NormBox[] {
	const rows = dims.reduce((acc, d) => acc * d, 1) / 16;
	if (!Number.isInteger(rows) || rows <= 0) return [];
	const boxes: NormBox[] = [];
	for (let r = 0; r < rows; r++) {
		const o = r * 16;
		// Row layout: top_y, top_x, bot_y, bot_x, then landmarks.
		const x = clamp01(data[o + 1]!);
		const y = clamp01(data[o]!);
		const x1 = clamp01(data[o + 3]!);
		const y1 = clamp01(data[o + 2]!);
		const w = x1 - x;
		const h = y1 - y;
		if (w <= 0.005 || h <= 0.005) continue;
		// Round to 5 decimals — float32 output noise would otherwise leak into
		// data-hb-reasons JSON and break exact-match assertions downstream.
		const round = (v: number) => Math.round(v * 1e5) / 1e5;
		boxes.push([round(x), round(y), round(w), round(h)]);
	}
	return boxes;
}

/**
 * Merge the per-pass box lists (index 0 = lowest confidence threshold) into
 * one scored list: a box earns the score of the highest pass it appears in.
 * Higher-pass boxes without a lower-pass match are kept at their own tier.
 */
export function mergeDetectionPasses(
	passes: readonly (readonly NormBox[])[],
): { box: NormBox; score: number }[] {
	const out: { box: NormBox; score: number }[] = [];
	for (let p = 0; p < passes.length; p++) {
		const score = FACE_DET_PASSES[p]?.score ?? 0;
		for (const box of passes[p] ?? []) {
			const hit = out.find((o) => boxIou(o.box, box) > FACE_MATCH_IOU);
			if (hit != null) {
				if (score > hit.score) {
					hit.score = score;
					hit.box = box;
				}
			} else {
				out.push({ box, score });
			}
		}
	}
	return out;
}

export function isFaceLabel(d: Detection): boolean {
	return d.label === "FACE_FEMALE" || d.label === "FACE_MALE";
}

/**
 * Ensemble-AND gate: a detector box becomes a synthetic detection only when
 * an independent NudeNet `FACE_*` detection of the same face exists. Either
 * model alone can false-positive — BlazeFace fires confidently on animal
 * faces (dog fixture: conf ≥0.8), NudeNet's gender is noisy — so additive
 * detections require both to localize the face. The payoff is score/box
 * quality: the synthetic carries the detector's tier score and tight box
 * instead of NudeNet's.
 */
export function corroboratedFaces(
	scored: readonly { box: NormBox; score: number }[],
	detections: readonly Detection[],
): { box: NormBox; score: number }[] {
	return scored.filter(({ box }) =>
		detections.some(
			(d) => isFaceLabel(d) && d.score >= FACE_CORROBORATE_MIN && sameFace(d.box, box),
		),
	);
}

/**
 * Drop NudeNet `FACE_*` detections that a dedicated-detector box covers —
 * the detector+FastFace pipeline's gender verdict replaces NudeNet's guess.
 * Non-overlapping NudeNet faces pass through untouched (they are the recall
 * fallback for faces the detector misses, e.g. artistic renditions).
 */
export function suppressCoveredFaceLabels(
	detections: readonly Detection[],
	faceBoxes: readonly NormBox[],
): Detection[] {
	if (faceBoxes.length === 0) return [...detections];
	return detections.filter((d) => {
		if (!isFaceLabel(d)) return true;
		return !faceBoxes.some((b) => sameFace(d.box, b));
	});
}

/**
 * Convert a `FACE_DET_SIZE²` RGBA buffer into the model's NCHW float32
 * tensor (RGB, 0..1 — this export is not ImageNet-normalized).
 */
export function blazeRgbaToTensor(rgba: Uint8ClampedArray): Float32Array {
	const plane = FACE_DET_SIZE * FACE_DET_SIZE;
	const out = new Float32Array(3 * plane);
	for (let i = 0; i < plane; i++) {
		const s = i * 4;
		out[i] = rgba[s]! / 255;
		out[plane + i] = rgba[s + 1]! / 255;
		out[2 * plane + i] = rgba[s + 2]! / 255;
	}
	return out;
}
