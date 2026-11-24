/**
 * YuNet face detector decode — pure math, zero ORT imports.
 *
 * Model: `face_detection_yunet_2023mar.onnx` (opencv_zoo, Apache-2.0,
 * ~232 KB, input [1,3,640,640] float32 BGR 0..255 — NO normalization, per
 * OpenCV `FaceDetectorYNImpl::detect` → `dnn::blobFromImage` defaults).
 *
 * Decode mirrors `modules/objdetect/src/face_detect.cpp::postProcess`:
 * per stride {8,16,32}, score = sqrt(clamp(cls)*clamp(obj)), box decoded
 * from (c+dx, r+dy)*stride + exp(w,h)*stride, 5 landmarks likewise, then
 * greedy NMS (IoU threshold, score-desc order).
 */

export const YUNET_INPUT_SIZE = 640;
export const YUNET_STRIDES = [8, 16, 32] as const;
export const YUNET_NMS_IOU = 0.3;
export const YUNET_SCORE_THRESHOLD = 0.6;

/** One decoded face in input-canvas pixel coordinates. */
export interface YuNetFace {
	/** x, y, w, h in input-canvas pixels. */
	box: { x: number; y: number; w: number; h: number };
	/** 5 landmarks in canvas pixels: right_eye, left_eye, nose, mouth_r, mouth_l. */
	landmarks: [number, number][];
	score: number;
}

/** Aspect-preserving letterbox of a srcW×srcH image into a square canvas. */
export interface Letterbox {
	/** Pixels of content per source pixel. */
	scale: number;
	/** Content origin inside the canvas. */
	ox: number;
	oy: number;
	/** Content size inside the canvas. */
	cw: number;
	ch: number;
}

export function letterboxRect(
	srcW: number,
	srcH: number,
	canvas: number = YUNET_INPUT_SIZE,
): Letterbox {
	const scale = Math.min(canvas / srcW, canvas / srcH);
	const cw = srcW * scale;
	const ch = srcH * scale;
	return { scale, ox: (canvas - cw) / 2, oy: (canvas - ch) / 2, cw, ch };
}

/** Map a canvas-pixel box back to normalized [0,1] source-image coords. */
export function unletterboxBox(
	box: { x: number; y: number; w: number; h: number },
	lb: Letterbox,
	srcW: number,
	srcH: number,
): { x: number; y: number; w: number; h: number } {
	return {
		x: (box.x - lb.ox) / lb.cw,
		y: (box.y - lb.oy) / lb.ch,
		w: box.w / lb.cw,
		h: box.h / lb.ch,
	};
}

export function unletterboxPoint(
	p: [number, number],
	lb: Letterbox,
	srcW: number,
	srcH: number,
): [number, number] {
	return [(p[0] - lb.ox) / lb.cw, (p[1] - lb.oy) / lb.ch];
}

export interface YuNetOutputs {
	cls_8: Float32Array;
	cls_16: Float32Array;
	cls_32: Float32Array;
	obj_8: Float32Array;
	obj_16: Float32Array;
	obj_32: Float32Array;
	bbox_8: Float32Array;
	bbox_16: Float32Array;
	bbox_32: Float32Array;
	kps_8: Float32Array;
	kps_16: Float32Array;
	kps_32: Float32Array;
}

const clamp01 = (n: number) => Math.min(1, Math.max(0, n));

function iou(
	a: { x: number; y: number; w: number; h: number },
	b: { x: number; y: number; w: number; h: number },
): number {
	const x1 = Math.max(a.x, b.x);
	const y1 = Math.max(a.y, b.y);
	const x2 = Math.min(a.x + a.w, b.x + b.w);
	const y2 = Math.min(a.y + a.h, b.y + b.h);
	const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
	const union = a.w * a.h + b.w * b.h - inter;
	return union <= 0 ? 0 : inter / union;
}

/**
 * Decode the 12 YuNet output tensors into faces in canvas-pixel coords.
 * `rows`/`cols` per stride derive from `inputSize` (pad to multiple of 32,
 * matching OpenCV's padWithDivisor — for the 640 model it is already exact).
 */
export function decodeYuNet(
	outputs: YuNetOutputs,
	opts: { inputSize?: number; scoreThreshold?: number; nmsThreshold?: number; topK?: number } = {},
): YuNetFace[] {
	const inputSize = opts.inputSize ?? YUNET_INPUT_SIZE;
	const scoreThreshold = opts.scoreThreshold ?? YUNET_SCORE_THRESHOLD;
	const nmsThreshold = opts.nmsThreshold ?? YUNET_NMS_IOU;
	const topK = opts.topK ?? 5000;

	// OpenCV pads input dims up to a multiple of 32 before decoding.
	const pad = Math.ceil(inputSize / 32) * 32;
	const byStride: Record<
		number,
		{ cls: Float32Array; obj: Float32Array; bbox: Float32Array; kps: Float32Array }
	> = {
		8: { cls: outputs.cls_8, obj: outputs.obj_8, bbox: outputs.bbox_8, kps: outputs.kps_8 },
		16: { cls: outputs.cls_16, obj: outputs.obj_16, bbox: outputs.bbox_16, kps: outputs.kps_16 },
		32: { cls: outputs.cls_32, obj: outputs.obj_32, bbox: outputs.bbox_32, kps: outputs.kps_32 },
	};

	const candidates: YuNetFace[] = [];
	for (const stride of YUNET_STRIDES) {
		const t = byStride[stride]!;
		const rows = Math.floor(pad / stride);
		const cols = Math.floor(pad / stride);
		for (let r = 0; r < rows; r++) {
			for (let c = 0; c < cols; c++) {
				const idx = r * cols + c;
				const score = Math.sqrt(clamp01(t.cls[idx] ?? 0) * clamp01(t.obj[idx] ?? 0));
				if (score < scoreThreshold) continue;
				const b = idx * 4;
				const cx = (c + (t.bbox[b] ?? 0)) * stride;
				const cy = (r + (t.bbox[b + 1] ?? 0)) * stride;
				const w = Math.exp(t.bbox[b + 2] ?? 0) * stride;
				const h = Math.exp(t.bbox[b + 3] ?? 0) * stride;
				const k = idx * 10;
				const landmarks = Array.from(
					{ length: 5 },
					(_, n) =>
						[
							((t.kps[k + 2 * n] ?? 0) + c) * stride,
							((t.kps[k + 2 * n + 1] ?? 0) + r) * stride,
						] as [number, number],
				);
				candidates.push({
					box: { x: cx - w / 2, y: cy - h / 2, w, h },
					landmarks,
					score,
				});
			}
		}
	}

	// Greedy NMS, score-desc — same order as OpenCV NMSBoxes.
	candidates.sort((a, b2) => b2.score - a.score);
	const kept: YuNetFace[] = [];
	for (const cand of candidates) {
		if (kept.length >= topK) break;
		if (kept.every((k) => iou(k.box, cand.box) <= nmsThreshold)) kept.push(cand);
	}
	return kept;
}

/**
 * Principled plausibility check on the 5 landmarks — rejects animal
 * snouts/pareidolia blobs without gating on NudeNet. Conservative: only
 * rejects clearly-degenerate geometry so profile/rotated faces pass.
 * Landmarks + box are in the same coordinate space.
 */
export function landmarksPlausible(face: YuNetFace): boolean {
	const { box, landmarks } = face;
	const [re, le, , rm, lm] = landmarks;
	if (re == null || le == null || rm == null || lm == null) return false;
	// Eyes must sit inside (or just outside) the box horizontally and above
	// the mouth line. Snout-like blobs put "eyes" at/below the mouth.
	const eyeMidY = (re[1] + le[1]) / 2;
	const mouthMidY = (rm[1] + lm[1]) / 2;
	if (eyeMidY > mouthMidY + box.h * 0.25) return false;
	// Degenerate inter-eye distance → not a real face.
	const eyeDist = Math.hypot(re[0] - le[0], re[1] - le[1]);
	if (eyeDist < box.w * 0.08 || eyeDist > box.w * 1.5) return false;
	// Landmarks must not lie wildly outside the box.
	const pad = Math.max(box.w, box.h) * 0.6;
	for (const [x, y] of landmarks) {
		if (x < box.x - pad || x > box.x + box.w + pad) return false;
		if (y < box.y - pad || y > box.y + box.h + pad) return false;
	}
	return true;
}
