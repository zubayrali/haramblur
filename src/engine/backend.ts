/**
 * onnxruntime-web session setup inside the engine worker (docs/PLAN.md "ML
 * runtime"): WebGPU when `navigator.gpu.requestAdapter()` succeeds, else WASM
 * (SIMD; threads only when cross-origin isolated).
 */
import * as ort from "onnxruntime-web";
import type { EngineBackend } from "../shared/protocol";
import { INPUT_SIZE } from "./nudenet";

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
}

export interface EngineSession {
	backend: EngineBackend;
	session: ort.InferenceSession;
	/**
	 * FastFace gender classifier session, always WASM: the model is tiny
	 * (~5 ms/inference) and a second WebGPU session would double the wedged-EP
	 * surface for no measurable gain.
	 */
	faceSession: ort.InferenceSession | null;
	/**
	 * YuNet face detector session, always WASM: ~232 KB model, single pass —
	 * a second WebGPU session would double the wedged-EP surface.
	 */
	yunetSession: ort.InferenceSession | null;
	/**
	 * nsfwjs MobileNetV2 whole-image classifier session, always WASM: ~2.8 MB
	 * model at 224² — a second WebGPU session would double the wedged-EP
	 * surface for no measurable gain.
	 */
	nsfwSession: ort.InferenceSession | null;
}

// Extension pages report `crossOriginIsolated === true`, so the threaded WASM
// build would spawn pthreads. Worker-in-worker pthread spawn inside Chrome
// offscreen documents is fragile — a failed spawn surfaces as an abort in the
// wasm glue and kills every inference. Single-threaded inference is ~110 ms
// here, so threads stay off regardless of what the context reports.
const WASM_THREADS = 1;

/**
 * Boot-time WebGPU probe budget. Some Chromium forks (observed: Helium)
 * vend a GPU adapter and create a session fine, then hang forever inside
 * `session.run` — the session reports ready while every inference stalls
 * until the host's 30 s timeout. One throwaway inference proves the EP
 * actually executes; slower-than-this means wedged, not merely slow.
 */
const WEBGPU_PROBE_TIMEOUT_MS = 5_000;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Run one inference on a zero tensor; resolves false on error or timeout. */
async function probeSession(session: ort.InferenceSession): Promise<boolean> {
	const inputName = session.inputNames[0];
	if (inputName == null) return false;
	const probe = new ort.Tensor("float32", new Float32Array(3 * INPUT_SIZE * INPUT_SIZE), [
		1,
		3,
		INPUT_SIZE,
		INPUT_SIZE,
	]);
	try {
		const outputs = await Promise.race([
			session.run({ [inputName]: probe }),
			sleep(WEBGPU_PROBE_TIMEOUT_MS).then(() => null),
		]);
		if (outputs == null) return false;
		for (const tensor of Object.values(outputs)) tensor.dispose();
		return true;
	} catch {
		return false;
	} finally {
		probe.dispose();
	}
}

/** Minimal WebGPU surface — the TS DOM lib in use lacks GPU typings. */
interface GpuLike {
	requestAdapter(): Promise<unknown>;
}

async function requestGpuAdapter(): Promise<unknown> {
	if (typeof navigator === "undefined") return null;
	const gpu = (navigator as { gpu?: GpuLike }).gpu;
	if (gpu == null) return null;
	try {
		return await gpu.requestAdapter();
	} catch {
		return null;
	}
}

/**
 * Configure `ort.env` and load the model. Tries the WebGPU EP first when a GPU
 * adapter is available; falls back to WASM when the adapter or the session
 * creation fails (the host maps that to `degraded`).
 */
export async function createSession({
	wasmBase,
	modelUrl,
	faceModelUrl,
	yunetModelUrl,
	nsfwModelUrl,
}: SessionOptions): Promise<EngineSession> {
	// Absolute extension URL — a relative path would resolve against the ort
	// module's own directory and 404 (docs/ENGINE_NOTES.md).
	ort.env.wasm.wasmPaths = wasmBase;
	ort.env.wasm.numThreads = WASM_THREADS;
	ort.env.logLevel = "warning";

	let backend: EngineBackend = "wasm";
	let session: ort.InferenceSession;
	if ((await requestGpuAdapter()) != null) {
		try {
			const candidate = await ort.InferenceSession.create(modelUrl, {
				executionProviders: ["webgpu"],
			});
			if (await probeSession(candidate)) {
				backend = "webgpu";
				session = candidate;
			} else {
				console.warn("[haramblur] WebGPU probe failed, falling back to WASM");
				await candidate.release();
				session = await ort.InferenceSession.create(modelUrl, {
					executionProviders: ["wasm"],
				});
			}
		} catch (err) {
			console.warn("[haramblur] WebGPU session failed, falling back to WASM:", err);
			session = await ort.InferenceSession.create(modelUrl, {
				executionProviders: ["wasm"],
			});
		}
	} else {
		session = await ort.InferenceSession.create(modelUrl, {
			executionProviders: ["wasm"],
		});
	}

	let faceSession: ort.InferenceSession | null = null;
	if (faceModelUrl != null) {
		try {
			faceSession = await ort.InferenceSession.create(faceModelUrl, {
				executionProviders: ["wasm"],
			});
		} catch (err) {
			// Missing/corrupt model must not take the detector down with it.
			console.warn("[haramblur] face gender model unavailable:", err);
		}
	}
	let yunetSession: ort.InferenceSession | null = null;
	if (yunetModelUrl != null) {
		try {
			yunetSession = await ort.InferenceSession.create(yunetModelUrl, {
				executionProviders: ["wasm"],
			});
		} catch (err) {
			console.warn("[haramblur] face detector model unavailable:", err);
		}
	}
	let nsfwSession: ort.InferenceSession | null = null;
	if (nsfwModelUrl != null) {
		try {
			nsfwSession = await ort.InferenceSession.create(nsfwModelUrl, {
				executionProviders: ["wasm"],
			});
		} catch (err) {
			console.warn("[haramblur] nsfw classifier model unavailable:", err);
		}
	}
	return { backend, session, faceSession, yunetSession, nsfwSession };
}
