/**
 * YOLOv8 instance-segmentation decode for person silhouettes.
 *
 * Tensor contract (verified with onnxruntime against the bundled export):
 * input `[1,3,640,640]` float16/float32 RGB 0..1 (read from session metadata), output0 `[1,4+classes+32,N]`
 * channel-major and output1 `[1,32,mh,mw]` mask prototypes. The class count
 * is derived from the tensor dimensions — never assumed — so a person-only
 * fine-tune (1 class rather than COCO's 80) decodes through the same path.
 * COCO's person class and person-only exports are both class index 0.
 */
import type { Detection, PersonMask } from "../shared/protocol";
import { nmsPerClass } from "./nudenet";

export const PERSON_INPUT_SIZE = 640;
export const PERSON_SCORE_FLOOR = 0.25;
export const PERSON_NMS_IOU = 0.5;
export const PERSON_MAX_INSTANCES = 4;
export const PERSON_CLASS_INDEX = 0;
/** Face-in-person containment required before falling back to raw IoU. */
export const PERSON_MIN_CONTAINMENT = 0.35;
/** Weak-overlap fallback for partially truncated detector boxes. */
export const PERSON_MIN_IOU = 0.02;
/** Longest side of a transported mask bitmap. */
export const MASK_MAX_DIM = 128;

/** Round one float32 value to IEEE-754 half-precision bits. */
export function float32ToFloat16Bits(value: number): number {
	if (Number.isNaN(value)) return 0x7e00;
	const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
	const v = Math.abs(value);
	if (v === 0) return sign;
	if (v >= 65520) return sign | 0x7c00;
	if (v < 2 ** -14) return sign | Math.min(0x3ff, Math.round(v / 2 ** -24));
	const exp = Math.floor(Math.log2(v));
	let exponent = exp + 15;
	let mantissa = Math.round((v / 2 ** exp - 1) * 1024);
	if (mantissa === 1024) {
		exponent++;
		mantissa = 0;
	}
	if (exponent >= 31) return sign | 0x7c00;
	return sign | (exponent << 10) | mantissa;
}

/** Expand IEEE-754 half-precision bits to float32. */
export function float16BitsToFloat32(bits: number): number {
	const sign = (bits & 0x8000) !== 0 ? -1 : 1;
	const exponent = (bits >> 10) & 0x1f;
	const mantissa = bits & 0x3ff;
	if (exponent === 0) return sign * mantissa * 2 ** -24;
	if (exponent === 31) return mantissa !== 0 ? Number.NaN : sign * Infinity;
	return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

/**
 * Copy tensor numeric data to float32. FP16 ORT tensors are `Uint16Array`
 * bit-storage where native `Float16Array` is unavailable; where it is
 * available the typed array already yields numeric values.
 */
export function tensorDataToFloat32(data: ArrayLike<number>): Float32Array {
	if (data instanceof Float32Array) return new Float32Array(data);
	const out = new Float32Array(data.length);
	if (data instanceof Uint16Array) {
		for (let i = 0; i < data.length; i++) out[i] = float16BitsToFloat32(data[i]!);
	} else {
		for (let i = 0; i < data.length; i++) out[i] = Number(data[i]);
	}
	return out;
}

export interface PersonInputSize {
	width: number;
	height: number;
}

export function personInputSizeFromShape(shape: readonly unknown[] | undefined): PersonInputSize {
	if (shape != null && shape.length === 4) {
		const h = shape[2];
		const w = shape[3];
		if (
			typeof h === "number" &&
			typeof w === "number" &&
			h >= 160 &&
			h <= 1280 &&
			w >= 160 &&
			w <= 1280
		) {
			return { width: w, height: h };
		}
	}
	return { width: PERSON_INPUT_SIZE, height: PERSON_INPUT_SIZE };
}

/** Centered Ultralytics-style letterbox geometry in model-input pixels. */
export interface PersonLetterbox {
	scale: number;
	dw: number;
	dh: number;
	padX: number;
	padY: number;
	inputW: number;
	inputH: number;
}

export function personLetterbox(
	srcW: number,
	srcH: number,
	input: PersonInputSize = { width: PERSON_INPUT_SIZE, height: PERSON_INPUT_SIZE },
): PersonLetterbox {
	const scale = Math.min(input.width / srcW, input.height / srcH);
	const dw = Math.max(1, Math.round(srcW * scale));
	const dh = Math.max(1, Math.round(srcH * scale));
	return {
		scale,
		dw,
		dh,
		padX: Math.round((input.width - dw) / 2 - 0.1) + 0,
		padY: Math.round((input.height - dh) / 2 - 0.1) + 0,
		inputW: input.width,
		inputH: input.height,
	};
}

export interface PersonInstance {
	/** Person-class confidence. */
	score: number;
	/** Clipped source-normalized `[x,y,w,h]` instance box. */
	box: Detection["box"];
	/** Clipped instance box in model-input pixels, xyxy. */
	inputBox: [x0: number, y0: number, x1: number, y1: number];
	/** Mask-prototype coefficients copied out of the detection tensor. */
	coeffs: Float32Array;
}

interface PersonCandidate {
	cls: number;
	score: number;
	x0: number;
	y0: number;
	x1: number;
	y1: number;
	index: number;
}

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

export interface DecodePersonsInput {
	detData: Float32Array;
	detRows: number;
	detCols: number;
	protoDims: readonly [maskDim: number, maskH: number, maskW: number];
	letterbox: PersonLetterbox;
	srcW: number;
	srcH: number;
	scoreFloor?: number;
	maxInstances?: number;
}

/** Decode output0 into person instances, reading class/mask counts from dims. */
export function decodePersons({
	detData,
	detRows,
	detCols,
	protoDims,
	letterbox,
	srcW,
	srcH,
	scoreFloor = PERSON_SCORE_FLOOR,
	maxInstances = PERSON_MAX_INSTANCES,
}: DecodePersonsInput): PersonInstance[] {
	const [maskDim] = protoDims;
	const numClasses = detRows - 4 - maskDim;
	if (numClasses <= 0) {
		throw new Error(`unexpected segmentation rows ${detRows} for ${maskDim} mask coefficients`);
	}
	if (PERSON_CLASS_INDEX >= numClasses) return [];
	const candidates: PersonCandidate[] = [];
	const personRow = 4 + PERSON_CLASS_INDEX;
	// Ultralytics' reference export emits xywh in input pixels, while some
	// deployment-oriented ONNX graphs (including the pinned EdgeFirst export)
	// emit the same tensor normalized to 0..1. Detect that once from the box
	// rows instead of hard-coding either contract.
	let maxCoord = 0;
	for (let row = 0; row < 4; row++) {
		for (let i = 0; i < detCols; i++) maxCoord = Math.max(maxCoord, detData[row * detCols + i]!);
	}
	const normalizedCoords = maxCoord <= 1.5;
	const sx = normalizedCoords ? letterbox.inputW : 1;
	const sy = normalizedCoords ? letterbox.inputH : 1;
	for (let i = 0; i < detCols; i++) {
		const score = detData[personRow * detCols + i]!;
		if (score < scoreFloor) continue;
		const cx = detData[i]! * sx;
		const cy = detData[detCols + i]! * sy;
		const w = detData[2 * detCols + i]! * sx;
		const h = detData[3 * detCols + i]! * sy;
		candidates.push({
			cls: PERSON_CLASS_INDEX,
			score,
			x0: cx - w / 2,
			y0: cy - h / 2,
			x1: cx + w / 2,
			y1: cy + h / 2,
			index: i,
		});
	}
	const contentX0 = letterbox.padX;
	const contentY0 = letterbox.padY;
	const contentX1 = letterbox.padX + letterbox.dw;
	const contentY1 = letterbox.padY + letterbox.dh;
	const coeffRow = 4 + numClasses;
	return nmsPerClass(candidates, PERSON_NMS_IOU)
		.sort((a, b) => b.score - a.score)
		.slice(0, maxInstances)
		.flatMap((c): PersonInstance[] => {
			const ix0 = Math.max(contentX0, c.x0);
			const iy0 = Math.max(contentY0, c.y0);
			const ix1 = Math.min(contentX1, c.x1);
			const iy1 = Math.min(contentY1, c.y1);
			if (ix1 <= ix0 || iy1 <= iy0) return [];
			const coeffs = new Float32Array(maskDim);
			for (let m = 0; m < maskDim; m++) coeffs[m] = detData[(coeffRow + m) * detCols + c.index]!;
			const x0 = clamp01((ix0 - letterbox.padX) / letterbox.scale / srcW);
			const y0 = clamp01((iy0 - letterbox.padY) / letterbox.scale / srcH);
			const x1 = clamp01((ix1 - letterbox.padX) / letterbox.scale / srcW);
			const y1 = clamp01((iy1 - letterbox.padY) / letterbox.scale / srcH);
			if (x1 <= x0 || y1 <= y0) return [];
			return [
				{
					score: c.score,
					box: [x0, y0, x1 - x0, y1 - y0],
					inputBox: [ix0, iy0, ix1, iy1],
					coeffs,
				},
			];
		});
}

export interface MaskRaster {
	width: number;
	height: number;
	rgba: Uint8ClampedArray;
	/** Inside-silhouette fraction of this bitmap. */
	coverage: number;
}

const clampInt = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, Math.round(n)));

function bilinear(
	logits: Float32Array,
	width: number,
	height: number,
	x: number,
	y: number,
): number {
	const cx = Math.min(width - 1, Math.max(0, x));
	const cy = Math.min(height - 1, Math.max(0, y));
	const x0 = Math.floor(cx);
	const y0 = Math.floor(cy);
	const x1 = Math.min(width - 1, x0 + 1);
	const y1 = Math.min(height - 1, y0 + 1);
	const fx = cx - x0;
	const fy = cy - y0;
	const p00 = logits[y0 * width + x0]!;
	const p10 = logits[y0 * width + x1]!;
	const p01 = logits[y1 * width + x0]!;
	const p11 = logits[y1 * width + x1]!;
	const top = p00 + (p10 - p00) * fx;
	const bottom = p01 + (p11 - p01) * fx;
	return top + (bottom - top) * fy;
}

/**
 * Compose one instance mask as a bounded RGBA bitmap covering exactly
 * `instance.inputBox`. Prototype logits are bilinearly sampled before the
 * zero threshold, approximating Ultralytics' upsample-then-crop path without
 * allocating a full source-resolution float mask.
 */
export function composeMaskRgba(
	instance: PersonInstance,
	protoData: Float32Array,
	protoDims: readonly [maskDim: number, maskH: number, maskW: number],
	inputW: number,
	inputH: number,
	maxDim = MASK_MAX_DIM,
): MaskRaster {
	const [maskDim, maskH, maskW] = protoDims;
	if (instance.coeffs.length !== maskDim) {
		throw new Error(`mask coeff count ${instance.coeffs.length} != prototype channels ${maskDim}`);
	}
	const [x0, y0, x1, y1] = instance.inputBox;
	const boxW = Math.max(1, x1 - x0);
	const boxH = Math.max(1, y1 - y0);
	// Two output pixels per prototype cell recover the bilinear edge without
	// pretending the model carries source-resolution detail.
	const protoBoxW = Math.max(1, (boxW * maskW) / inputW);
	const protoBoxH = Math.max(1, (boxH * maskH) / inputH);
	const width = clampInt(protoBoxW * 2, 8, maxDim);
	const height = clampInt(protoBoxH * 2, 8, maxDim);
	const plane = maskW * maskH;
	const logits = new Float32Array(plane);
	for (let c = 0; c < maskDim; c++) {
		const coeff = instance.coeffs[c]!;
		if (coeff === 0) continue;
		const base = c * plane;
		for (let i = 0; i < plane; i++) logits[i]! += coeff * protoData[base + i]!;
	}
	const rgba = new Uint8ClampedArray(width * height * 4);
	let inside = 0;
	const protoXPerInput = maskW / inputW;
	const protoYPerInput = maskH / inputH;
	for (let oy = 0; oy < height; oy++) {
		const inputY = y0 + ((oy + 0.5) / height) * boxH;
		const py = (inputY + 0.5) * protoYPerInput - 0.5;
		for (let ox = 0; ox < width; ox++) {
			const inputX = x0 + ((ox + 0.5) / width) * boxW;
			const px = (inputX + 0.5) * protoXPerInput - 0.5;
			if (bilinear(logits, maskW, maskH, px, py) > 0) {
				inside++;
				const o = (oy * width + ox) * 4;
				rgba[o] = 255;
				rgba[o + 1] = 255;
				rgba[o + 2] = 255;
				rgba[o + 3] = 255;
			}
		}
	}
	return { width, height, rgba, coverage: inside / (width * height) };
}

/** Intersection area as a fraction of box `a` (the face). */
export function boxContainment(
	a: readonly [number, number, number, number],
	b: readonly [number, number, number, number],
): number {
	const ix0 = Math.max(a[0], b[0]);
	const iy0 = Math.max(a[1], b[1]);
	const ix1 = Math.min(a[0] + a[2], b[0] + b[2]);
	const iy1 = Math.min(a[1] + a[3], b[1] + b[3]);
	const inter = Math.max(0, ix1 - ix0) * Math.max(0, iy1 - iy0);
	const area = a[2] * a[3];
	return area > 0 ? inter / area : 0;
}

function iouBox(
	a: readonly [number, number, number, number],
	b: readonly [number, number, number, number],
): number {
	const ix0 = Math.max(a[0], b[0]);
	const iy0 = Math.max(a[1], b[1]);
	const ix1 = Math.min(a[0] + a[2], b[0] + b[2]);
	const iy1 = Math.min(a[1] + a[3], b[1] + b[3]);
	const inter = Math.max(0, ix1 - ix0) * Math.max(0, iy1 - iy0);
	if (inter <= 0) return 0;
	const union = a[2] * a[3] + b[2] * b[3] - inter;
	return union > 0 ? inter / union : 0;
}

export interface PersonAssociation {
	/** Index into the input face array. */
	faceIndex: number;
	/** Index into the input person array. */
	personIndex: number;
	containment: number;
	iou: number;
}

/**
 * Associate female face boxes with person instances. Containment is primary
 * because face/person boxes have intrinsically tiny IoU on full-body people;
 * raw IoU is only a weak fallback for truncated boxes. Multiple faces may
 * share one person; every face independently chooses its best instance.
 */
export function associateFacesToPersons(
	faces: readonly (readonly [number, number, number, number])[],
	persons: readonly PersonInstance[],
	minContainment = PERSON_MIN_CONTAINMENT,
	minIou = PERSON_MIN_IOU,
): PersonAssociation[] {
	const out: PersonAssociation[] = [];
	for (let faceIndex = 0; faceIndex < faces.length; faceIndex++) {
		const face = faces[faceIndex]!;
		let best: { personIndex: number; containment: number; iou: number; score: number } | null =
			null;
		for (let personIndex = 0; personIndex < persons.length; personIndex++) {
			const person = persons[personIndex]!;
			const candidate = {
				personIndex,
				containment: boxContainment(face, person.box),
				iou: iouBox(face, person.box),
				score: person.score,
			};
			if (
				best == null ||
				candidate.containment > best.containment + 1e-9 ||
				(Math.abs(candidate.containment - best.containment) <= 1e-9 &&
					(candidate.iou > best.iou + 1e-9 ||
						(Math.abs(candidate.iou - best.iou) <= 1e-9 && candidate.score > best.score)))
			) {
				best = candidate;
			}
		}
		if (best != null && (best.containment >= minContainment || best.iou >= minIou)) {
			out.push({
				faceIndex,
				personIndex: best.personIndex,
				containment: best.containment,
				iou: best.iou,
			});
		}
	}
	return out;
}

/** Wire-format mask metadata once the raster has been encoded by the host. */
export function wireMask(
	instance: PersonInstance,
	width: number,
	height: number,
	bits: string,
	coverage: number,
): PersonMask {
	return { box: instance.box, width, height, bits, coverage };
}
