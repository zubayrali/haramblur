/**
 * Engine worker: onnxruntime-web inference in a dedicated Worker spawned by
 * the engine host (docs/PLAN.md). Receives an `init` message carrying the
 * absolute extension URLs for the ORT WASM assets and the model, then `infer`
 * messages carrying a transferable ImageBitmap.
 */
import * as ort from "onnxruntime-web";
import type { Detection, EngineBackend } from "../shared/protocol";
import { createSession } from "./backend";
import {
	blazeRgbaToTensor,
	corroboratedFaces,
	FACE_DET_IOU,
	FACE_DET_MAX,
	FACE_DET_PASSES,
	FACE_DET_SIZE,
	mergeDetectionPasses,
	type NormBox,
	parseFaceBoxes,
	suppressCoveredFaceLabels,
} from "./facedet";
import {
	FACE_INPUT_SIZE,
	FACE_MAX_PER_PASS,
	FACE_MIN_PX,
	type FaceClassifier,
	type FaceGender,
	faceCropRect,
	faceRgbaToTensor,
	genderProbs,
	refineFaceDetections,
} from "./fastface";
import {
	INPUT_SIZE,
	LETTERBOX_FILL,
	letterboxSize,
	parseDetections,
	rgbaToTensor,
	tileRects,
} from "./nudenet";

export interface WorkerInitMessage {
	type: "init";
	wasmBase: string;
	modelUrl: string;
	/** Optional FastFace gender model; absent → FACE_* labels stay NudeNet's. */
	faceModelUrl?: string;
	/** Optional BlazeFace detector; absent → NudeNet FACE_* labels untouched. */
	faceDetModelUrl?: string;
}

export interface WorkerInferMessage {
	type: "infer";
	id: number;
	bitmap: ImageBitmap;
	/**
	 * Detail pass (WP7): run only the tiled crops instead of the whole-image
	 * pass. The host requests it when the whole pass decided `clear`, so small
	 * faces crushed by the 320 letterbox get a second look at higher zoom.
	 */
	detail?: boolean;
}

export type WorkerInMessage = WorkerInitMessage | WorkerInferMessage;

export type WorkerOutMessage =
	| { type: "ready"; backend: EngineBackend }
	| { type: "initError"; error: string }
	| { type: "result"; id: number; detections: Detection[]; ms: number }
	| { type: "inferError"; id: number; error: string };

const scope = self as unknown as {
	onmessage: ((event: MessageEvent<WorkerInMessage>) => void) | null;
	postMessage(message: WorkerOutMessage, transfer?: Transferable[]): void;
};

let session: ort.InferenceSession | null = null;
let faceSession: ort.InferenceSession | null = null;
let faceDetSession: ort.InferenceSession | null = null;
const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
const ctx = canvas.getContext("2d");
const faceCanvas = new OffscreenCanvas(FACE_INPUT_SIZE, FACE_INPUT_SIZE);
const faceCtx = faceCanvas.getContext("2d", { willReadFrequently: true });
const faceDetCanvas = new OffscreenCanvas(FACE_DET_SIZE, FACE_DET_SIZE);
const faceDetCtx = faceDetCanvas.getContext("2d", { willReadFrequently: true });

/**
 * Inference runs are serialized: onnxruntime-web does not guarantee
 * concurrent `session.run` safety on one session, and the WebGPU EP wedges
 * under parallel calls (observed: adapter + session fine, runs hang — the
 * engine reports ready while every request stalls). Queue fetch/decode
 * still overlaps in the host; only the GPU/WASM call itself is serialized.
 */
let inferChain: Promise<void> = Promise.resolve();

const errMsg = (err: unknown) => (err instanceof Error ? err.message : String(err));

function letterboxDraw(bitmap: ImageBitmap): { rgba: Uint8ClampedArray; k: number } {
	return letterboxDrawRegion(bitmap, 0, 0, bitmap.width, bitmap.height);
}

/** Letterbox a source-pixel rect of the bitmap into the square input. */
function letterboxDrawRegion(
	bitmap: ImageBitmap,
	rx: number,
	ry: number,
	rw: number,
	rh: number,
): { rgba: Uint8ClampedArray; k: number } {
	if (ctx == null) throw new Error("OffscreenCanvas 2d context unavailable");
	const { k, dw, dh } = letterboxSize(rw, rh);
	ctx.fillStyle = `rgb(${LETTERBOX_FILL},${LETTERBOX_FILL},${LETTERBOX_FILL})`;
	ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
	ctx.drawImage(bitmap, rx, ry, rw, rh, 0, 0, dw, dh);
	const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
	return { rgba: data, k };
}

async function runTensor(rgba: Uint8ClampedArray): Promise<{
	data: Float32Array;
	rows: number;
	cols: number;
}> {
	const input = new ort.Tensor("float32", rgbaToTensor(rgba), [1, 3, INPUT_SIZE, INPUT_SIZE]);
	const inputName = session!.inputNames[0];
	if (inputName == null) throw new Error("model exposes no input");
	const outputs = await session!.run({ [inputName]: input });
	const output = outputs[session!.outputNames[0]!];
	if (output == null) throw new Error("model produced no output");
	const [batch, rows, cols] = output.dims;
	if (batch !== 1 || rows == null || cols == null || rows < 4) {
		throw new Error(`unexpected output shape ${JSON.stringify(output.dims)}`);
	}
	return { data: output.data as Float32Array, rows, cols };
}

/**
 * Crop one normalized face box out of the bitmap, run FastFace, and return
 * gender probabilities. Returns null when the session is missing, the crop
 * is degenerate, or the model output is malformed — callers keep NudeNet's
 * label in that case.
 */
function makeFaceClassifier(bitmap: ImageBitmap): FaceClassifier {
	return async (box): Promise<FaceGender | null> => {
		if (faceSession == null || faceCtx == null) return null;
		const rect = faceCropRect(box, bitmap.width, bitmap.height);
		if (rect == null || rect.w < FACE_MIN_PX || rect.h < FACE_MIN_PX) return null;
		faceCtx.clearRect(0, 0, FACE_INPUT_SIZE, FACE_INPUT_SIZE);
		faceCtx.imageSmoothingEnabled = true;
		faceCtx.imageSmoothingQuality = "high";
		faceCtx.drawImage(
			bitmap,
			rect.x,
			rect.y,
			rect.w,
			rect.h,
			0,
			0,
			FACE_INPUT_SIZE,
			FACE_INPUT_SIZE,
		);
		const { data } = faceCtx.getImageData(0, 0, FACE_INPUT_SIZE, FACE_INPUT_SIZE);
		const inputName = faceSession.inputNames[0];
		if (inputName == null) return null;
		const tensor = new ort.Tensor("float32", faceRgbaToTensor(data), [
			1,
			3,
			FACE_INPUT_SIZE,
			FACE_INPUT_SIZE,
		]);
		try {
			const outputs = await faceSession.run({ [inputName]: tensor });
			const logits = (outputs["gender_logits"]?.data ??
				outputs[faceSession.outputNames[0]!]?.data) as Float32Array | undefined;
			const ageOut = (outputs["age"]?.data ?? outputs[faceSession.outputNames[2]!]?.data) as
				| Float32Array
				| undefined;
			if (logits == null) return null;
			const { female, male } = genderProbs(logits);
			return {
				femaleProb: female,
				maleProb: male,
				age: ageOut?.[0] ?? 0,
			};
		} finally {
			tensor.dispose();
		}
	};
}

/**
 * Run the fused BlazeFace export at each confidence pass and merge the
 * results into scored boxes. The graph has no score output, so a box's
 * score is the tier of the highest `conf_threshold` it survives.
 */
async function detectFaceBoxes(bitmap: ImageBitmap): Promise<{ box: NormBox; score: number }[]> {
	if (faceDetSession == null || faceDetCtx == null) return [];
	faceDetCtx.clearRect(0, 0, FACE_DET_SIZE, FACE_DET_SIZE);
	faceDetCtx.drawImage(bitmap, 0, 0, FACE_DET_SIZE, FACE_DET_SIZE);
	const { data } = faceDetCtx.getImageData(0, 0, FACE_DET_SIZE, FACE_DET_SIZE);
	const tensor = new ort.Tensor("float32", blazeRgbaToTensor(data), [
		1,
		3,
		FACE_DET_SIZE,
		FACE_DET_SIZE,
	]);
	const inputName = faceDetSession.inputNames[0];
	if (inputName == null) {
		tensor.dispose();
		return [];
	}
	const passes: NormBox[][] = [];
	try {
		for (const pass of FACE_DET_PASSES) {
			const outputs = await faceDetSession.run({
				[inputName]: tensor,
				conf_threshold: new ort.Tensor("float32", new Float32Array([pass.conf]), [1]),
				iou_threshold: new ort.Tensor("float32", new Float32Array([FACE_DET_IOU]), [1]),
				max_detections: new ort.Tensor("int64", new BigInt64Array([BigInt(FACE_DET_MAX)]), [1]),
			});
			const t = outputs["selectedBoxes"] ?? outputs[faceDetSession.outputNames[0]!];
			passes.push(t == null ? [] : parseFaceBoxes(t.data as ArrayLike<number>, t.dims as number[]));
			for (const o of Object.values(outputs)) o.dispose();
		}
	} finally {
		tensor.dispose();
	}
	return mergeDetectionPasses(passes);
}

/** FastFace gender for each detected face box → synthetic FACE_* detections. */
async function genderizeFaceBoxes(
	scored: readonly { box: NormBox; score: number }[],
	bitmap: ImageBitmap,
): Promise<Detection[]> {
	const classify = makeFaceClassifier(bitmap);
	const out: Detection[] = [];
	for (const { box, score } of scored.slice(0, FACE_MAX_PER_PASS)) {
		const [bw, bh] = [box[2] * bitmap.width, box[3] * bitmap.height];
		if (bw < FACE_MIN_PX || bh < FACE_MIN_PX) continue;
		const gender = await classify(box).catch(() => null);
		if (gender == null) continue;
		const conf = Math.max(gender.femaleProb, gender.maleProb);
		// Unlike refineFaceDetections there is no NudeNet label to fall back
		// on — an ambiguous crop is dropped rather than guessed.
		if (conf < 0.6) continue;
		out.push({
			label: gender.femaleProb >= gender.maleProb ? "FACE_FEMALE" : "FACE_MALE",
			score,
			box,
		});
	}
	return out;
}

async function infer(id: number, bitmap: ImageBitmap, detail: boolean): Promise<void> {
	if (session == null) {
		scope.postMessage({ type: "inferError", id, error: "engine not initialized" });
		return;
	}
	const t0 = performance.now();
	try {
		let detections: Detection[] = [];
		let faceDets: Detection[] = [];
		if (detail) {
			// Detail pass: only tiles, remapped back to source-normalized boxes.
			for (const rect of tileRects(bitmap.width, bitmap.height)) {
				const { rgba, k } = letterboxDrawRegion(bitmap, rect.x, rect.y, rect.w, rect.h);
				const { data, rows, cols } = await runTensor(rgba);
				for (const d of parseDetections(data, rows, cols, k, rect.w, rect.h)) {
					const [x, y, w, h] = d.box;
					detections.push({
						label: d.label,
						score: d.score,
						box: [
							(rect.x + x * rect.w) / bitmap.width,
							(rect.y + y * rect.h) / bitmap.height,
							(w * rect.w) / bitmap.width,
							(h * rect.h) / bitmap.height,
						],
					});
				}
			}
		} else {
			const { rgba, k } = letterboxDraw(bitmap);
			const { data, rows, cols } = await runTensor(rgba);
			detections.push(...parseDetections(data, rows, cols, k, bitmap.width, bitmap.height));
			if (faceDetSession != null && faceSession != null) {
				// Dedicated face pipeline: BlazeFace boxes → FastFace gender.
				// Synthetics are gated on NudeNet face corroboration — the
				// detector alone false-positives on animal faces. NudeNet
				// FACE_* labels overlapping an emitted face are suppressed:
				// the specialist verdict replaces the guess.
				const scored = await detectFaceBoxes(bitmap);
				faceDets = await genderizeFaceBoxes(corroboratedFaces(scored, detections), bitmap);
				if (faceDets.length > 0) {
					detections = suppressCoveredFaceLabels(
						detections,
						faceDets.map((d) => d.box),
					);
				}
			}
		}
		if (faceSession != null) {
			detections = await refineFaceDetections(
				detections,
				makeFaceClassifier(bitmap),
				bitmap.width,
				bitmap.height,
			);
		}
		// Synthetic detections are final — appended after refinement so
		// FastFace doesn't re-classify its own output.
		detections.push(...faceDets);
		detections.sort((a, b) => b.score - a.score);
		scope.postMessage({ type: "result", id, detections, ms: performance.now() - t0 });
	} catch (err) {
		scope.postMessage({ type: "inferError", id, error: errMsg(err) });
	} finally {
		bitmap.close();
	}
}

scope.onmessage = (event) => {
	const msg = event.data;
	if (msg.type === "init") {
		createSession({
			wasmBase: msg.wasmBase,
			modelUrl: msg.modelUrl,
			faceModelUrl: msg.faceModelUrl,
			faceDetModelUrl: msg.faceDetModelUrl,
		})
			.then(({ backend, session: s, faceSession: fs, faceDetSession: fds }) => {
				session = s;
				faceSession = fs;
				faceDetSession = fds;
				scope.postMessage({ type: "ready", backend });
			})
			.catch((err) => scope.postMessage({ type: "initError", error: errMsg(err) }));
	} else if (msg.type === "infer") {
		inferChain = inferChain
			.then(() => infer(msg.id, msg.bitmap, msg.detail === true))
			.catch(() => {
				// infer() reports failures via inferError; keep the chain alive
				// so one bad request can't stall every later inference.
			});
	}
};
