/**
 * NudeNet `320n.onnx` preprocessing and output parsing (docs/ENGINE_NOTES.md).
 *
 * - Input `[1, 3, 320, 320]` float32 RGB 0..1, letterboxed top-left with fill
 *   `rgb(114,114,114)` and scale `k = min(320/w, 320/h)`.
 * - Output `[1, 4 + 18, N]` channel-major: rows 0-3 are `cx, cy, w, h` in input
 *   pixels, rows 4.. are class scores. Divide by `k` to map back, apply
 *   per-class NMS at IoU 0.45, return boxes normalized 0..1 over the source.
 */
import { NUDENET_LABELS } from "../shared/labels";
import type { Detection } from "../shared/protocol";

export const INPUT_SIZE = 320;
export const LETTERBOX_FILL = 114;
export const NMS_IOU = 0.45;
/**
 * Candidates below this score are dropped before NMS. Below every threshold
 * the policy can produce (strictness 1 → exposed 0.15, faces 0.2).
 */
export const SCORE_FLOOR = 0.1;
/**
 * Detail pass (WP7): when the whole-image pass finds nothing the policy would
 * blur, a tiled second pass re-runs the model on overlapping crops so faces
 * letterboxed below the model's floor (~20 input px) land at detectable size.
 * Grid grows with source size, capped at `MAX_TILES_PER_AXIS`² tiles.
 */
export const TILE_OVERLAP = 0.25;
export const MAX_TILES_PER_AXIS = 3;

export interface Letterbox {
	/** Uniform scale `min(input/w, input/h)`. */
	k: number;
	/** Drawn width/height inside the square input. */
	dw: number;
	dh: number;
}

/** Letterbox geometry for a source of `srcW × srcH` — top-left anchored. */
export function letterboxSize(srcW: number, srcH: number, input = INPUT_SIZE): Letterbox {
	const k = Math.min(input / srcW, input / srcH);
	return { k, dw: Math.round(srcW * k), dh: Math.round(srcH * k) };
}

/** A source-pixel rectangle covered by one detail-pass tile. */
export interface TileRect {
	x: number;
	y: number;
	w: number;
	h: number;
}

const clampInt = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, Math.round(n)));

/** Positions of `n` tiles covering `[0, len]` with `overlap` fractional overlap. */
function tileAxis(len: number, n: number, overlap: number): TileRect[] {
	if (n <= 1) return [{ x: 0, y: 0, w: len, h: 0 }];
	const span = (len * (1 + overlap * (n - 1))) / n;
	const step = (len - span) / (n - 1);
	const rects: TileRect[] = [];
	for (let i = 0; i < n; i++) {
		const x = clampInt(i * step, 0, Math.max(0, len - span));
		rects.push({ x, y: 0, w: clampInt(Math.min(span, len - x), 1, len), h: 0 });
	}
	return rects;
}

/**
 * Overlapping source rects for the detail pass. Tile count per axis is
 * `ceil(dim × 1.5 / input)` capped at `MAX_TILES_PER_AXIS`, so tiles zoom in
 * ~1.5–4.5× over the whole-image scale while staying bounded (≤ 9 model calls
 * per detail pass). Empty for images that fit the input on both axes.
 */
export function tileRects(
	srcW: number,
	srcH: number,
	input = INPUT_SIZE,
	overlap = TILE_OVERLAP,
): TileRect[] {
	// `1.5×` density: a 480px source gets 3 tiles/axis (~240px each → ~1.33
	// zoom vs the 0.67 whole-image scale) instead of two ~300px tiles (~1.07),
	// which barely helps faces already near the model floor.
	const cols = clampInt(Math.ceil((srcW * 1.5) / input), 1, MAX_TILES_PER_AXIS);
	const rows = clampInt(Math.ceil((srcH * 1.5) / input), 1, MAX_TILES_PER_AXIS);
	if (cols === 1 && rows === 1) return [];
	const xs = tileAxis(srcW, cols, overlap);
	const ys = tileAxis(srcH, rows, overlap);
	const rects: TileRect[] = [];
	for (const ry of ys) {
		for (const rx of xs) {
			rects.push({ x: rx.x, y: ry.x, w: rx.w, h: ry.w });
		}
	}
	return rects;
}

/**
 * Merge detections from several passes. Boxes are already normalized over the
 * same source; same-class pairs overlapping above `NMS_IOU` keep the higher
 * score (e.g. a face seen in two adjacent tiles). Returns score-descending.
 */
export function mergeDetections(...lists: readonly Detection[][]): Detection[] {
	const candidates = lists.flat().flatMap((d) => {
		const cls = NUDENET_LABELS.indexOf(d.label);
		if (cls < 0) return [];
		const [x, y, w, h] = d.box;
		return [{ cls, score: d.score, x0: x, y0: y, x1: x + w, y1: y + h }];
	});
	return nmsPerClass(candidates, NMS_IOU)
		.map(
			(c): Detection => ({
				label: NUDENET_LABELS[c.cls]!,
				score: c.score,
				box: [c.x0, c.y0, c.x1 - c.x0, c.y1 - c.y0],
			}),
		)
		.sort((a, b) => b.score - a.score);
}

/**
 * Convert the letterboxed `input × input` RGBA buffer into the model's CHW
 * float32 tensor (RGB, 0..1). The buffer must already contain the drawn image
 * at (0, 0) plus the 114 fill padding.
 */
export function rgbaToTensor(rgba: Uint8ClampedArray, input = INPUT_SIZE): Float32Array {
	const plane = input * input;
	const out = new Float32Array(3 * plane);
	for (let i = 0; i < plane; i++) {
		const s = i * 4;
		out[i] = rgba[s]! / 255;
		out[plane + i] = rgba[s + 1]! / 255;
		out[2 * plane + i] = rgba[s + 2]! / 255;
	}
	return out;
}

/** A detection candidate in input-pixel coordinates. */
interface Candidate {
	cls: number;
	score: number;
	x0: number;
	y0: number;
	x1: number;
	y1: number;
}

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

function iou(a: Candidate, b: Candidate): number {
	const ix0 = Math.max(a.x0, b.x0);
	const iy0 = Math.max(a.y0, b.y0);
	const ix1 = Math.min(a.x1, b.x1);
	const iy1 = Math.min(a.y1, b.y1);
	const iw = Math.max(0, ix1 - ix0);
	const ih = Math.max(0, iy1 - iy0);
	const inter = iw * ih;
	const union = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter;
	return union > 0 ? inter / union : 0;
}

/** Per-class greedy NMS: keep the highest-score box of each IoU > threshold group. */
export function nmsPerClass<T extends Candidate>(
	candidates: readonly T[],
	iouThreshold = NMS_IOU,
): T[] {
	const byClass = new Map<number, T[]>();
	for (const c of candidates) {
		const list = byClass.get(c.cls);
		if (list) list.push(c);
		else byClass.set(c.cls, [c]);
	}
	const kept: T[] = [];
	for (const list of byClass.values()) {
		list.sort((a, b) => b.score - a.score);
		const keptOfClass: T[] = [];
		for (const c of list) {
			if (keptOfClass.every((k) => iou(c, k) <= iouThreshold)) keptOfClass.push(c);
		}
		kept.push(...keptOfClass);
	}
	return kept;
}

/**
 * Parse the `[1, 4 + classes, N]` output tensor into normalized detections.
 * `k` is the letterbox scale and `srcW`/`srcH` the source pixels.
 */
export function parseDetections(
	data: Float32Array,
	rows: number,
	cols: number,
	k: number,
	srcW: number,
	srcH: number,
	scoreFloor = SCORE_FLOOR,
): Detection[] {
	const numClasses = Math.min(rows - 4, NUDENET_LABELS.length);
	const candidates: Candidate[] = [];
	for (let i = 0; i < cols; i++) {
		let best = 0;
		let bestScore = Number.NEGATIVE_INFINITY;
		for (let c = 0; c < numClasses; c++) {
			const s = data[(4 + c) * cols + i]!;
			if (s > bestScore) {
				bestScore = s;
				best = c;
			}
		}
		if (bestScore < scoreFloor) continue;
		const cx = data[i]!;
		const cy = data[cols + i]!;
		const w = data[2 * cols + i]!;
		const h = data[3 * cols + i]!;
		candidates.push({
			cls: best,
			score: bestScore,
			x0: cx - w / 2,
			y0: cy - h / 2,
			x1: cx + w / 2,
			y1: cy + h / 2,
		});
	}
	return nmsPerClass(candidates, NMS_IOU)
		.map((c): Detection => {
			const x0 = clamp01(c.x0 / k / srcW);
			const y0 = clamp01(c.y0 / k / srcH);
			const x1 = clamp01(c.x1 / k / srcW);
			const y1 = clamp01(c.y1 / k / srcH);
			return {
				// cls < numClasses ≤ NUDENET_LABELS.length, so this is always defined.
				label: NUDENET_LABELS[c.cls]!,
				score: c.score,
				box: [x0, y0, x1 - x0, y1 - y0],
			};
		})
		.sort((a, b) => b.score - a.score);
}
