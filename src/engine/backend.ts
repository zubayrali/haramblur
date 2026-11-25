/**
 * onnxruntime-web session setup inside the engine worker (docs/PLAN.md "ML
 * runtime"). Backend selection is per model: every session tries WebGPU when
 * an adapter is reachable, proves execution with a zero-tensor warm-up, and
 * falls back to WASM independently (docs/DETECTION_PARITY.md DP1).
 */
import * as ort from "onnxruntime-web";
import type { ModelSlot } from "../shared/model-slots";
import type { EngineBackend } from "../shared/protocol";

export interface SessionOptions {
	/** Absolute extension URL of the shipped ORT WASM assets (`/ort/`). */
	wasmBase: string;
	/** Absolute extension URL of `models/320n.onnx`. */
	modelUrl: string;
	/** Absolute extension URL of `models/fastface-small-112.onnx` (optional). */
	faceModelUrl?: string;
	/** Absolute extension URL of `models/yunet-2023mar.onnx` (optional). */
	yunetModelUrl?: string;
	/** Absolute extension URL of `models/nsfwjs-u8.onnx` (optional). */
	nsfwModelUrl?: string;
	/** Absolute extension URL of `models/yolov8n-seg-fp16.onnx` (optional). */
	personModelUrl?: string;
	/** FP32 person-segmentation fallback used when adapter `shader-f16` is absent. */
	personModelFp32Url?: string;
	/** OPFS-stored override weights per slot — win over bundled URLs. */
	modelOverrides?: Partial<Record<ModelSlot, Uint8Array>>;
	/** Internal test/verification escape hatch; never user-facing. */
	forceWasm?: boolean;
}

export interface EngineSession {
	/** NudeNet backend, retained as the engine-level health/degraded signal. */
	backend: EngineBackend;
	/** Actual backend chosen independently for each loaded model slot. */
	backends: Partial<Record<ModelSlot, EngineBackend>>;
	session: ort.InferenceSession;
	faceSession: ort.InferenceSession | null;
	yunetSession: ort.InferenceSession | null;
	nsfwSession: ort.InferenceSession | null;
	personSession: ort.InferenceSession | null;
}

// Worker-in-worker pthread spawn inside Chrome offscreen documents is fragile.
const WASM_THREADS = 1;
const PROBE_INPUT_SIZE: Record<ModelSlot, number> = {
	nudenet: 320,
	fastface: 112,
	yunet: 640,
	nsfw: 224,
	person: 640,
};
/** A wedged EP must fail the boot warm-up rather than stall inference forever. */
const SESSION_PROBE_TIMEOUT_MS = 5_000;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

type TensorData = Float32Array | Uint16Array;

interface InputMetadataLike {
	isTensor?: boolean;
	type?: string;
	shape?: readonly unknown[];
}

function inputDimensions(
	shape: readonly unknown[] | undefined,
	probeInputSize: number,
): number[] | null {
	if (shape == null || shape.length === 0) return null;
	const dims = shape.map((dim, index) => {
		if (typeof dim === "number" && dim > 0) return Math.floor(dim);
		// ORT reports symbolic batch/spatial dimensions as strings. Batch is 1;
		// symbolic spatial dimensions use the slot's packaged input size.
		return index === 0 ? 1 : probeInputSize;
	});
	return dims.every((dim) => Number.isFinite(dim) && dim > 0) ? dims : null;
}

/** Construct a zero input from ORT metadata; symbolic dimensions become one. */
export function zeroInputTensor(
	session: ort.InferenceSession,
	probeInputSize: number,
): ort.Tensor | null {
	const meta = session.inputMetadata[0] as InputMetadataLike | undefined;
	if (meta?.isTensor !== true) return null;
	const dims = inputDimensions(meta.shape, probeInputSize);
	if (dims == null) return null;
	const size = dims.reduce((product, dim) => product * dim, 1);
	// A metadata bug must not allocate unbounded memory during boot.
	if (!Number.isSafeInteger(size) || size <= 0 || size > 20_000_000) return null;
	let data: TensorData;
	if (meta.type === "float16") data = new Uint16Array(size);
	else if (meta.type === "float32") data = new Float32Array(size);
	else return null;
	return new ort.Tensor(meta.type, data, dims);
}

/** Run one zero-tensor inference; resolves false on error, timeout, or no input. */
export async function probeSession(
	session: ort.InferenceSession,
	probeInputSize: number,
): Promise<boolean> {
	const inputName = session.inputNames[0];
	const input = zeroInputTensor(session, probeInputSize);
	if (inputName == null || input == null) return false;
	let outputs: Record<string, ort.Tensor> | null | undefined;
	try {
		const runPromise = session.run({ [inputName]: input });
		outputs = await Promise.race([runPromise, sleep(SESSION_PROBE_TIMEOUT_MS).then(() => null)]);
		if (outputs == null) {
			void runPromise
				.then((late) => {
					for (const tensor of Object.values(late)) tensor.dispose();
				})
				.catch(() => undefined);
			return false;
		}
		for (const tensor of Object.values(outputs)) tensor.dispose();
		return true;
	} catch {
		return false;
	} finally {
		input.dispose();
	}
}

/** Minimal WebGPU surface — the TS DOM lib in use lacks GPU typings. */
interface GpuAdapterLike {
	features?: { has?(feature: string): boolean };
}
interface GpuLike {
	requestAdapter(): Promise<GpuAdapterLike | null>;
}

async function requestGpuAdapter(): Promise<GpuAdapterLike | null> {
	if (typeof navigator === "undefined") return null;
	const gpu = (navigator as { gpu?: GpuLike }).gpu;
	if (gpu == null) return null;
	try {
		return await gpu.requestAdapter();
	} catch {
		return null;
	}
}

export function adapterSupportsShaderF16(adapter: GpuAdapterLike | null): boolean {
	return adapter?.features?.has?.("shader-f16") === true;
}

interface LoadedModel {
	session: ort.InferenceSession;
	backend: EngineBackend;
}

/**
 * Configure `ort.env` and load every model independently. A WebGPU failure in
 * one model neither promotes another model to WebGPU nor demotes it to WASM.
 */
export async function createSession({
	wasmBase,
	modelUrl,
	faceModelUrl,
	yunetModelUrl,
	nsfwModelUrl,
	personModelUrl,
	personModelFp32Url,
	modelOverrides,
	forceWasm = false,
}: SessionOptions): Promise<EngineSession> {
	ort.env.wasm.wasmPaths = wasmBase;
	ort.env.wasm.numThreads = WASM_THREADS;
	ort.env.logLevel = "warning";

	const ov = modelOverrides ?? {};
	const createModel = (src: string | Uint8Array, executionProviders: EngineBackend[]) =>
		typeof src === "string"
			? ort.InferenceSession.create(src, { executionProviders })
			: ort.InferenceSession.create(new Uint8Array(src), { executionProviders });
	const adapter = forceWasm ? null : await requestGpuAdapter();
	// ORT WASM memory and WebGPU shader compilation are not safe to stress with
	// five concurrent first-runs in one worker. Session creation may overlap;
	// zero-tensor execution remains serialized.
	let probeChain: Promise<unknown> = Promise.resolve();
	const queuedProbe = (session: ort.InferenceSession, slot: ModelSlot) => {
		const result = probeChain.then(() => probeSession(session, PROBE_INPUT_SIZE[slot]));
		probeChain = result.catch(() => undefined);
		return result;
	};

	async function loadModel(slot: ModelSlot, src: string | Uint8Array): Promise<LoadedModel> {
		if (adapter != null) {
			try {
				const candidate = await createModel(src, ["webgpu"]);
				if (await queuedProbe(candidate, slot)) {
					return { session: candidate, backend: "webgpu" };
				}
				console.warn(`[haramblur] ${slot} WebGPU warm-up failed, falling back to WASM`);
				await candidate.release().catch(() => undefined);
			} catch (err) {
				console.warn(`[haramblur] ${slot} WebGPU session failed, falling back to WASM:`, err);
			}
		}
		const session = await createModel(src, ["wasm"]);
		if (!(await queuedProbe(session, slot))) {
			await session.release().catch(() => undefined);
			throw new Error(`${slot} WASM warm-up failed`);
		}
		return { session, backend: "wasm" };
	}

	async function loadOptional(slot: ModelSlot, src: string | Uint8Array | undefined) {
		if (src == null) return null;
		try {
			return await loadModel(slot, src);
		} catch (err) {
			console.warn(`[haramblur] ${slot} model unavailable:`, err);
			return null;
		}
	}

	// The fp16 segmentation pack is the default because it is half the size.
	// Without adapter shader-f16, prefer the fp32 pack for both EPs rather than
	// asking WebGPU or WASM to execute a dtype the runtime/context may not support.
	const personBundled =
		adapterSupportsShaderF16(adapter) || personModelFp32Url == null
			? personModelUrl
			: personModelFp32Url;

	// Model sessions are independent; parallel boot keeps the DP1 warm-up inside
	// the 2 s cold-start budget instead of summing five shader-compile latencies.
	const [nudenet, fastface, yunet, nsfw, person] = await Promise.all([
		loadModel("nudenet", ov.nudenet ?? modelUrl),
		loadOptional("fastface", ov.fastface ?? faceModelUrl),
		loadOptional("yunet", ov.yunet ?? yunetModelUrl),
		loadOptional("nsfw", ov.nsfw ?? nsfwModelUrl),
		loadOptional("person", ov.person ?? personBundled),
	]);
	const backends: Partial<Record<ModelSlot, EngineBackend>> = { nudenet: nudenet.backend };
	if (fastface != null) backends.fastface = fastface.backend;
	if (yunet != null) backends.yunet = yunet.backend;
	if (nsfw != null) backends.nsfw = nsfw.backend;
	if (person != null) backends.person = person.backend;

	console.info("[haramblur] model backends:", backends);
	return {
		backend: nudenet.backend,
		backends,
		session: nudenet.session,
		faceSession: fastface?.session ?? null,
		yunetSession: yunet?.session ?? null,
		nsfwSession: nsfw?.session ?? null,
		personSession: person?.session ?? null,
	};
}
