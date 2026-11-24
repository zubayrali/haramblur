/**
 * FastFace `fastface-small-112.onnx` face gender refinement (Apache-2.0).
 *
 * NudeNet's `FACE_FEMALE`/`FACE_MALE` labels are weak — eval shows male
 * portraits scoring `FACE_FEMALE@0.8+` while `FACE_MALE` recall sits near
 * 33%. FastFace is a dedicated MobileNetV3 gender+age classifier (~94%
 * balanced accuracy on public benchmarks) that expects a square crop of a
 * detected face. We reuse NudeNet's face *boxes* (its gender label is the
 * only unreliable part), crop each unique face, and let FastFace correct
 * the gender — but only when it is confident; unsure cases keep NudeNet's
 * original label so recall never regresses.
 *
 * Tensor contract (verified against the upstream inference code):
 * - Input `[1, 3, 112, 112]` float32, RGB, ImageNet-normalized
 *   (mean 0.485/0.456/0.406, std 0.229/0.224/0.225), NCHW.
 * - Output `gender_logits [1, 2]` — index 0 = female, 1 = male (softmax).
 *   `age` is a scalar expected-age estimate (unused for now).
 */
import type { Detection } from "../shared/protocol";

export const FACE_INPUT_SIZE = 112;
/** Box expansion on each side before cropping (upstream uses 0.2). */
export const FACE_CROP_MARGIN = 0.25;
/** Boxes smaller than this many source pixels on either axis are skipped —
 * upscaling them to 112px produces noise, not signal. */
export const FACE_MIN_PX = 16;
/** Cap on FastFace runs per inference pass (group photos can have many faces). */
export const FACE_MAX_PER_PASS = 8;
/** FastFace must be at least this confident to override NudeNet's label. */
export const FACE_RELABEL_MIN_CONF = 0.7;
/** Face boxes (either gender label) overlapping above this IoU are the
 * same face — NudeNet sometimes emits both labels on one box. */
export const FACE_CLUSTER_IOU = 0.3;

export const FACE_LABELS = new Set<Detection["label"]>(["FACE_FEMALE", "FACE_MALE", "FACE_PERSON"]);

export interface FaceGender {
	femaleProb: number;
	maleProb: number;
	/** Estimated age; absent when the model's age output is unavailable. */
	age?: number;
}

/**
 * Classify one face. `box` is the normalized `[x, y, w, h]` detection box;
 * `landmarks` (optional, normalized) are the detector's 5-point landmarks —
 * when present the crop is eye-aligned instead of axis-aligned. Implementations
 * crop from the source bitmap, run the model, and return probabilities — or
 * `null` when the crop is unusable.
 */
export type FaceClassifier = (
	box: readonly [number, number, number, number],
	landmarks?: readonly (readonly [number, number])[],
) => Promise<FaceGender | null>;

const toEdges = (box: readonly [number, number, number, number]) => ({
	x0: box[0],
	y0: box[1],
	x1: box[0] + box[2],
	y1: box[1] + box[3],
});

export function boxIou(
	a: readonly [number, number, number, number],
	b: readonly [number, number, number, number],
): number {
	const ea = toEdges(a);
	const eb = toEdges(b);
	const iw = Math.max(0, Math.min(ea.x1, eb.x1) - Math.max(ea.x0, eb.x0));
	const ih = Math.max(0, Math.min(ea.y1, eb.y1) - Math.max(ea.y0, eb.y0));
	const inter = iw * ih;
	if (inter <= 0) return 0;
	const union = (ea.x1 - ea.x0) * (ea.y1 - ea.y0) + (eb.x1 - eb.x0) * (eb.y1 - eb.y0) - inter;
	return union > 0 ? inter / union : 0;
}

interface FaceCluster {
	/** Representative detection (highest NudeNet score in the cluster). */
	rep: Detection;
	/** Every FACE_* detection belonging to this face. */
	members: Detection[];
}

/**
 * Group FACE_* detections into unique faces. NudeNet can emit both
 * `FACE_FEMALE` and `FACE_MALE` on the same box; they must share one
 * FastFace call and one output detection.
 */
function clusterFaces(detections: readonly Detection[]): FaceCluster[] {
	const faces = detections
		.filter((d) => FACE_LABELS.has(d.label))
		.sort((a, b) => b.score - a.score);
	const clusters: FaceCluster[] = [];
	for (const d of faces) {
		const hit = clusters.find((c) => boxIou(c.rep.box, d.box) > FACE_CLUSTER_IOU);
		if (hit != null) hit.members.push(d);
		else clusters.push({ rep: d, members: [d] });
	}
	return clusters;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/**
 * Source-pixel crop rect for a normalized face box, expanded by
 * `FACE_CROP_MARGIN`. Returns null when the box is degenerate.
 */
export function faceCropRect(
	box: readonly [number, number, number, number],
	srcW: number,
	srcH: number,
): { x: number; y: number; w: number; h: number } | null {
	const [nx, ny, nw, nh] = box;
	const mx = nw * FACE_CROP_MARGIN;
	const my = nh * FACE_CROP_MARGIN;
	const x = clamp((nx - mx) * srcW, 0, srcW);
	const y = clamp((ny - my) * srcH, 0, srcH);
	const x1 = clamp((nx + nw + mx) * srcW, 0, srcW);
	const y1 = clamp((ny + nh + my) * srcH, 0, srcH);
	const w = x1 - x;
	const h = y1 - y;
	if (w < 1 || h < 1) return null;
	return { x, y, w, h };
}

/**
 * Re-label FACE_* detections using a dedicated gender classifier.
 *
 * For each unique face (deduped across both gender labels, capped at
 * `FACE_MAX_PER_PASS`, skipping sub-`FACE_MIN_PX` boxes):
 * - classifier confident (≥ `FACE_RELABEL_MIN_CONF`) → the cluster collapses
 *   to a single detection with the corrected `FACE_FEMALE`/`FACE_MALE`
 *   label, keeping the representative's NudeNet score (that score measures
 *   "is there a face here", which FastFace cannot judge);
 * - classifier not confident (< `FACE_RELABEL_MIN_CONF`) → the cluster
 *   collapses to a neutral `FACE_PERSON` — the face is real but its gender
 *   isn't verified, so NudeNet's noisy gendered label is suppressed without
 *   asserting one (eval: NudeNet FACE_FEMALE fires on men at ~0.5);
 * - classifier unavailable (crop unusable, call failed) → the cluster's
 *   original detections pass through unchanged.
 *
 * Non-face detections are untouched.
 */
export async function refineFaceDetections(
	detections: readonly Detection[],
	classify: FaceClassifier,
	srcW: number,
	srcH: number,
): Promise<Detection[]> {
	const clusters = clusterFaces(detections);
	if (clusters.length === 0) return [...detections];

	// member → replacement detection, or null to drop (collapsed duplicate).
	const replacements = new Map<Detection, Detection | null>();
	for (const cluster of clusters.slice(0, FACE_MAX_PER_PASS)) {
		const rep = cluster.rep;
		const [bw, bh] = [rep.box[2] * srcW, rep.box[3] * srcH];
		if (bw < FACE_MIN_PX || bh < FACE_MIN_PX) continue;
		const gender = await classify(rep.box).catch(() => null);
		if (gender == null) continue;
		const conf = Math.max(gender.femaleProb, gender.maleProb);
		if (conf >= FACE_RELABEL_MIN_CONF) {
			const label: Detection["label"] =
				gender.femaleProb >= gender.maleProb ? "FACE_FEMALE" : "FACE_MALE";
			for (const m of cluster.members) {
				replacements.set(
					m,
					m === rep ? { label, score: rep.score, box: rep.box, age: gender.age } : null,
				);
			}
		} else {
			// Gender not verified: collapse the cluster to a neutral
			// placeholder. NudeNet's FACE_FEMALE is noisy on male faces —
			// keeping it would blur men; asserting a gender FastFace can't
			// support is worse. Covers the whole sub-RELABEL band, including
			// near-coin-flip outputs (~0.5) that previously passed through.
			for (const m of cluster.members) {
				replacements.set(
					m,
					m === rep
						? { label: "FACE_PERSON", score: rep.score, box: rep.box, age: gender.age }
						: null,
				);
			}
		}
	}

	return detections.flatMap((d) => {
		if (!replacements.has(d)) return [d];
		const r = replacements.get(d);
		return r == null ? [] : [r];
	});
}

/**
 * Similarity transform that eye-aligns a detected face into an
 * `outSize`² canvas: rotates the eye-line horizontal, centers the eye
 * midpoint at (50%, 35%), and scales so the inter-eye distance is ~32% of
 * the crop. `landmarks` are normalized source coords in detector order
 * (right_eye, left_eye, nose, mouth_r, mouth_l) — only the two eyes are
 * used. Returns canvas `setTransform` params, or null when degenerate.
 */
export function faceAlignTransform(
	landmarks: readonly (readonly [number, number])[],
	srcW: number,
	srcH: number,
	outSize: number = FACE_INPUT_SIZE,
): { a: number; b: number; c: number; d: number; e: number; f: number } | null {
	const re = landmarks[0];
	const le = landmarks[1];
	if (re == null || le == null) return null;
	const ex = ((re[0] + le[0]) / 2) * srcW;
	const ey = ((re[1] + le[1]) / 2) * srcH;
	const dx = (le[0] - re[0]) * srcW;
	const dy = (le[1] - re[1]) * srcH;
	const eyeDist = Math.hypot(dx, dy);
	if (eyeDist < 2) return null;
	const theta = Math.atan2(dy, dx);
	const cos = Math.cos(theta);
	const sin = Math.sin(theta);
	const k = Math.min(8, Math.max(0.25, (0.32 * outSize) / eyeDist));
	// p' = k·R(-θ)·(p - E) + T, T = (S/2, 0.35·S)
	const a = k * cos;
	const c = k * sin;
	const b = -k * sin;
	const d = k * cos;
	const tx = outSize / 2;
	const ty = 0.35 * outSize;
	return { a, b, c, d, e: tx - (a * ex + c * ey), f: ty - (b * ex + d * ey) };
}

const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
const IMAGENET_STD = [0.229, 0.224, 0.225] as const;

/**
 * Convert an `FACE_INPUT_SIZE²` RGBA buffer into the model's NCHW float32
 * tensor (RGB 0..1, ImageNet-normalized).
 */
export function faceRgbaToTensor(rgba: Uint8ClampedArray): Float32Array {
	const plane = FACE_INPUT_SIZE * FACE_INPUT_SIZE;
	const out = new Float32Array(3 * plane);
	for (let i = 0; i < plane; i++) {
		const s = i * 4;
		for (let c = 0; c < 3; c++) {
			out[c * plane + i] = (rgba[s + c]! / 255 - IMAGENET_MEAN[c]!) / IMAGENET_STD[c]!;
		}
	}
	return out;
}

/** Softmax over the 2-class gender logits → probabilities. */
export function genderProbs(logits: ArrayLike<number>): { female: number; male: number } {
	const e0 = Math.exp(logits[0] ?? 0);
	const e1 = Math.exp(logits[1] ?? 0);
	const sum = e0 + e1;
	return { female: e0 / sum, male: e1 / sum };
}
