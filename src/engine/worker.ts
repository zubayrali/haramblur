/**
 * Engine worker: onnxruntime-web inference in a dedicated Worker spawned by
 * the engine host (docs/PLAN.md). Receives an `init` message carrying the
 * absolute extension URLs for the ORT WASM assets and the model, then `infer`
 * messages carrying a transferable ImageBitmap.
 */
import * as ort from "onnxruntime-web";
import { decide } from "../shared/policy";
import type { Detection, EngineBackend } from "../shared/protocol";
import type { Settings } from "../shared/settings";
import { createSession } from "./backend";
import {
	boxIou,
	FACE_CLUSTER_IOU,
	FACE_INPUT_SIZE,
	FACE_LABELS,
	FACE_MIN_PX,
	type FaceClassifier,
	type FaceGender,
	faceAlignTransform,
	faceCropRect,
	faceRgbaToTensor,
	genderProbs,
	refineFaceDetections,
} from "./fastface";
import { NSFW_INPUT_SIZE, nsfwDetection, nsfwVerdict, softmax } from "./nsfw";
import {
	INPUT_SIZE,
	LETTERBOX_FILL,
	letterboxSize,
	parseDetections,
	rgbaToTensor,
	tileRects,
} from "./nudenet";
import {
	decodeYuNet,
	landmarksPlausible,
	letterboxRect,
	unletterboxBox,
	unletterboxPoint,
	YUNET_INPUT_SIZE,
} from "./yunet";

export interface WorkerInitMessage {
	type: "init";
	wasmBase: string;
	modelUrl: string;
	/** Optional FastFace gender model; absent → FACE_* labels stay NudeNet's. */
	faceModelUrl?: string;
	/** Optional YuNet face detector; absent → no dedicated face pass. */
	yunetModelUrl?: string;
	/** Optional nsfwjs whole-image classifier; absent → no NSFW signal. */
	nsfwModelUrl?: string;
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
	/**
	 * Resolved settings for this request — used to skip the NSFW pass when
	 * the detections already decide `blur` (the verdict can't change).
	 */
	settings?: Settings;
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
let yunetSession: ort.InferenceSession | null = null;
let nsfwSession: ort.InferenceSession | null = null;
const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
const ctx = canvas.getContext("2d");
const faceCanvas = new OffscreenCanvas(FACE_INPUT_SIZE, FACE_INPUT_SIZE);
const faceCtx = faceCanvas.getContext("2d", { willReadFrequently: true });
const yunetCanvas = new OffscreenCanvas(YUNET_INPUT_SIZE, YUNET_INPUT_SIZE);
const yunetCtx = yunetCanvas.getContext("2d", { willReadFrequently: true });
const nsfwCanvas = new OffscreenCanvas(NSFW_INPUT_SIZE, NSFW_INPUT_SIZE);
const nsfwCtx = nsfwCanvas.getContext("2d", { willReadFrequently: true });
/** YuNet faces per inference pass — group photos can have many. */
const YUNET_MAX_PER_PASS = 12;
/** FastFace must be this confident to emit a YuNet synthetic detection. */
const YUNET_GENDER_MIN_CONF = 0.85;

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
	let outputs: Record<string, ort.Tensor> | undefined;
	try {
		outputs = await session!.run({ [inputName]: input });
		const output = outputs[session!.outputNames[0]!];
		if (output == null) throw new Error("model produced no output");
		const [batch, rows, cols] = output.dims;
		if (batch !== 1 || rows == null || cols == null || rows < 4) {
			throw new Error(`unexpected output shape ${JSON.stringify(output.dims)}`);
		}
		// Copy before dispose — tensor buffers are recycled by ORT.
		return { data: new Float32Array(output.data as Float32Array), rows, cols };
	} finally {
		input.dispose();
		if (outputs != null) {
			for (const t of Object.values(outputs)) t.dispose();
		}
	}
}

/**
 * Crop one normalized face box out of the bitmap, run FastFace, and return
 * gender probabilities. Returns null when the session is missing, the crop
 * is degenerate, or the model output is malformed — callers keep NudeNet's
 * label in that case.
 */
function makeFaceClassifier(bitmap: ImageBitmap): FaceClassifier {
	return async (box, landmarks): Promise<FaceGender | null> => {
		if (faceSession == null || faceCtx == null) return null;
		faceCtx.clearRect(0, 0, FACE_INPUT_SIZE, FACE_INPUT_SIZE);
		faceCtx.imageSmoothingEnabled = true;
		faceCtx.imageSmoothingQuality = "high";
		const aligned =
			landmarks != null && landmarks.length >= 2
				? faceAlignTransform(landmarks, bitmap.width, bitmap.height)
				: null;
		if (aligned != null) {
			// Eye-aligned similarity transform over the whole bitmap; only the
			// face region lands inside the 112² viewport.
			faceCtx.setTransform(aligned.a, aligned.b, aligned.c, aligned.d, aligned.e, aligned.f);
			faceCtx.drawImage(bitmap, 0, 0);
			faceCtx.setTransform(1, 0, 0, 1, 0, 0);
		} else {
			const rect = faceCropRect(box, bitmap.width, bitmap.height);
			if (rect == null || rect.w < FACE_MIN_PX || rect.h < FACE_MIN_PX) return null;
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
		}
		const { data } = faceCtx.getImageData(0, 0, FACE_INPUT_SIZE, FACE_INPUT_SIZE);
		const inputName = faceSession.inputNames[0];
		if (inputName == null) return null;
		const tensor = new ort.Tensor("float32", faceRgbaToTensor(data), [
			1,
			3,
			FACE_INPUT_SIZE,
			FACE_INPUT_SIZE,
		]);
		let outputs: Record<string, ort.Tensor> | undefined;
		try {
			outputs = await faceSession.run({ [inputName]: tensor });
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
				age: ageOut?.[0],
			};
		} finally {
			tensor.dispose();
			if (outputs != null) {
				for (const t of Object.values(outputs)) t.dispose();
			}
		}
	};
}

interface YuNetFaceNorm {
	box: [number, number, number, number];
	landmarks: [number, number][];
	score: number;
}

/**
 * Letterbox the bitmap into the 640² YuNet canvas (aspect preserved, black
 * pad — matching OpenCV's BORDER_CONSTANT), convert RGBA→BGR float32 NCHW
 * (raw 0..255, no normalization), run the session, and decode to
 * source-normalized boxes + landmarks. Returns [] when the session is
 * absent. Tensors are disposed before returning.
 */
async function detectYuNetFaces(bitmap: ImageBitmap): Promise<YuNetFaceNorm[]> {
	if (yunetSession == null || yunetCtx == null) return [];
	const lb = letterboxRect(bitmap.width, bitmap.height);
	yunetCtx.fillStyle = "#000";
	yunetCtx.fillRect(0, 0, YUNET_INPUT_SIZE, YUNET_INPUT_SIZE);
	yunetCtx.drawImage(bitmap, lb.ox, lb.oy, lb.cw, lb.ch);
	const { data: rgba } = yunetCtx.getImageData(0, 0, YUNET_INPUT_SIZE, YUNET_INPUT_SIZE);
	const plane = YUNET_INPUT_SIZE * YUNET_INPUT_SIZE;
	const tensorData = new Float32Array(3 * plane);
	for (let i = 0; i < plane; i++) {
		const s = i * 4;
		// BGR order, raw 0..255 — YuNet's blobFromImage uses no normalization.
		tensorData[i] = rgba[s + 2]!;
		tensorData[plane + i] = rgba[s + 1]!;
		tensorData[2 * plane + i] = rgba[s]!;
	}
	const inputName = yunetSession.inputNames[0];
	if (inputName == null) return [];
	const input = new ort.Tensor("float32", tensorData, [1, 3, YUNET_INPUT_SIZE, YUNET_INPUT_SIZE]);
	let outputs: Record<string, ort.Tensor> | undefined;
	try {
		outputs = await yunetSession.run({ [inputName]: input });
		const get = (name: string) => outputs?.[name]?.data as Float32Array | undefined;
		const decoded = decodeYuNet({
			cls_8: get("cls_8") ?? new Float32Array(0),
			cls_16: get("cls_16") ?? new Float32Array(0),
			cls_32: get("cls_32") ?? new Float32Array(0),
			obj_8: get("obj_8") ?? new Float32Array(0),
			obj_16: get("obj_16") ?? new Float32Array(0),
			obj_32: get("obj_32") ?? new Float32Array(0),
			bbox_8: get("bbox_8") ?? new Float32Array(0),
			bbox_16: get("bbox_16") ?? new Float32Array(0),
			bbox_32: get("bbox_32") ?? new Float32Array(0),
			kps_8: get("kps_8") ?? new Float32Array(0),
			kps_16: get("kps_16") ?? new Float32Array(0),
			kps_32: get("kps_32") ?? new Float32Array(0),
		});
		return decoded
			.filter(landmarksPlausible)
			.map((f) => {
				const b = unletterboxBox(f.box, lb, bitmap.width, bitmap.height);
				return {
					box: [b.x, b.y, b.w, b.h] as [number, number, number, number],
					landmarks: f.landmarks.map((p) => unletterboxPoint(p, lb, bitmap.width, bitmap.height)),
					score: f.score,
				};
			})
			.filter(
				(f) =>
					f.box[0] + f.box[2] > 0 &&
					f.box[1] + f.box[3] > 0 &&
					f.box[0] < 1 &&
					f.box[1] < 1 &&
					f.box[2] > 0 &&
					f.box[3] > 0,
			);
	} finally {
		input.dispose();
		if (outputs != null) {
			for (const t of Object.values(outputs)) t.dispose();
		}
	}
}

/**
 * Whole-image nsfwjs pass: stretch the bitmap to 224² (the model was trained
 * on stretched resizes — aspect preservation would shift its composition
 * priors), convert RGBA→RGB float32 0..1 NHWC, and reduce the 5-class output
 * to an NSFW margin detection. Returns null when the session is absent or
 * the image does not lean NSFW. Whole pass only — detail tiles would repeat
 * context the whole pass already saw.
 */
async function classifyNsfw(bitmap: ImageBitmap): Promise<Detection | null> {
	if (nsfwSession == null || nsfwCtx == null) return null;
	nsfwCtx.drawImage(bitmap, 0, 0, NSFW_INPUT_SIZE, NSFW_INPUT_SIZE);
	const { data: rgba } = nsfwCtx.getImageData(0, 0, NSFW_INPUT_SIZE, NSFW_INPUT_SIZE);
	const plane = NSFW_INPUT_SIZE * NSFW_INPUT_SIZE;
	const tensorData = new Float32Array(3 * plane);
	for (let i = 0; i < plane; i++) {
		const s = i * 4;
		tensorData[i] = rgba[s]! / 255;
		tensorData[plane + i] = rgba[s + 1]! / 255;
		tensorData[2 * plane + i] = rgba[s + 2]! / 255;
	}
	const inputName = nsfwSession.inputNames[0];
	if (inputName == null) return null;
	const input = new ort.Tensor("float32", tensorData, [1, NSFW_INPUT_SIZE, NSFW_INPUT_SIZE, 3]);
	let outputs: Record<string, ort.Tensor> | undefined;
	try {
		outputs = await nsfwSession.run({ [inputName]: input });
		const out = outputs[nsfwSession.outputNames[0]!];
		if (out == null) return null;
		const probs = softmax(out.data as Float32Array);
		const verdict = nsfwVerdict(probs);
		return verdict == null ? null : nsfwDetection(verdict);
	} finally {
		input.dispose();
		if (outputs != null) {
			for (const t of Object.values(outputs)) t.dispose();
		}
	}
}

async function infer(
	id: number,
	bitmap: ImageBitmap,
	detail: boolean,
	settings: Settings | undefined,
): Promise<void> {
	if (session == null) {
		scope.postMessage({ type: "inferError", id, error: "engine not initialized" });
		return;
	}
	const t0 = performance.now();
	try {
		let detections: Detection[] = [];
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
		}
		const classifier = faceSession != null ? makeFaceClassifier(bitmap) : null;
		// YuNet face pass (whole-image only): detect → genderize with
		// landmark-aligned FastFace → suppress NudeNet FACE_* labels covered by
		// CONFIDENT synthetics → refine leftover NudeNet faces → append
		// synthetics. Requires BOTH sessions — without FastFace there is no
		// gender signal for YuNet boxes.
		let yunetFaces: YuNetFaceNorm[] = [];
		const yunetChecked = !detail && yunetSession != null && classifier != null;
		if (yunetChecked) {
			yunetFaces = await detectYuNetFaces(bitmap);
		}
		if (classifier != null) {
			const synthetics: Detection[] = [];
			for (const f of yunetFaces.slice(0, YUNET_MAX_PER_PASS)) {
				const g = await classifier(f.box, f.landmarks).catch(() => null);
				if (g == null) continue;
				const conf = Math.max(g.femaleProb, g.maleProb);
				if (conf >= YUNET_GENDER_MIN_CONF) {
					synthetics.push({
						label: g.femaleProb >= g.maleProb ? "FACE_FEMALE" : "FACE_MALE",
						score: f.score * conf,
						box: f.box,
						...(g.age != null ? { age: g.age } : {}),
					});
				} else {
					// Gender unverified: emit the neutral placeholder so the IoU
					// filter below still suppresses NudeNet's noisy FACE_* on this
					// box — without asserting one (same rule as refineFaceDetections).
					synthetics.push({ label: "FACE_PERSON", score: f.score * conf, box: f.box });
				}
			}
			if (synthetics.length > 0) {
				detections = detections.filter(
					(d) =>
						!FACE_LABELS.has(d.label) ||
						!synthetics.some((s) => boxIou(s.box, d.box) > FACE_CLUSTER_IOU),
				);
			}
			detections = await refineFaceDetections(detections, classifier, bitmap.width, bitmap.height);
			// NudeNet FACE_* labels that no YuNet face corroborates are usually
			// mislabels (beards, hands, fabric reading as faces). When YuNet ran
			// — even finding zero faces — demote them to the neutral placeholder.
			if (yunetChecked) {
				detections = detections.map((d) =>
					FACE_LABELS.has(d.label) &&
					d.label !== "FACE_PERSON" &&
					!yunetFaces.some((f) => boxIou(f.box, d.box) > FACE_CLUSTER_IOU)
						? { ...d, label: "FACE_PERSON" as Detection["label"] }
						: d,
				);
			}
			detections.push(...synthetics);
		}
		// Whole-image NSFW pass, last: it exists to catch suggestive/anime
		// content NudeNet misses, so it only runs when the detections so far
		// don't already decide `blur` — UNLESS region blur is on, where an
		// NSFW verdict's full-frame box still escalates to whole-element blur
		// (a face-only verdict must not trap a bikini shot in region mode).
		// Detail tiles never run it — whole-image context was seen.
		if (!detail && nsfwSession != null) {
			const alreadyBlur = settings != null && decide(detections, settings) === "blur";
			// Missing settings (tests/old hosts) → run it: can't prove blur.
			if (!alreadyBlur || settings?.regionBlur !== false) {
				const nsfw = await classifyNsfw(bitmap);
				if (nsfw != null) detections.push(nsfw);
			}
		}
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
			yunetModelUrl: msg.yunetModelUrl,
			nsfwModelUrl: msg.nsfwModelUrl,
		})
			.then(({ backend, session: s, faceSession: fs, yunetSession: ys, nsfwSession: ns }) => {
				session = s;
				faceSession = fs;
				yunetSession = ys;
				nsfwSession = ns;
				scope.postMessage({ type: "ready", backend });
			})
			.catch((err) => scope.postMessage({ type: "initError", error: errMsg(err) }));
	} else if (msg.type === "infer") {
		inferChain = inferChain
			.then(() => infer(msg.id, msg.bitmap, msg.detail === true, msg.settings))
			.catch(() => {
				// infer() reports failures via inferError; keep the chain alive
				// so one bad request can't stall every later inference.
			});
	}
};
