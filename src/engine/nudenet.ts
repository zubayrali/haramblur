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
