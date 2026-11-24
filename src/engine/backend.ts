/**
 * onnxruntime-web session setup inside the engine worker (docs/PLAN.md "ML
 * runtime"): WebGPU when `navigator.gpu.requestAdapter()` succeeds, else WASM
 * (SIMD; threads only when cross-origin isolated).
 */
import * as ort from "onnxruntime-web";
import type { EngineBackend } from "../shared/protocol";

export interface SessionOptions {
	/** Absolute extension URL of the shipped ORT WASM assets (`/ort/`). */
	wasmBase: string;
	/** Absolute extension URL of `models/320n.onnx`. */
	modelUrl: string;
}

export interface EngineSession {
	backend: EngineBackend;
	session: ort.InferenceSession;
}

// Extension pages report `crossOriginIsolated === true`, so the threaded WASM
// build would spawn pthreads. Worker-in-worker pthread spawn inside Chrome
// offscreen documents is fragile — a failed spawn surfaces as an abort in the
// wasm glue and kills every inference. Single-threaded inference is ~110 ms
// here, so threads stay off regardless of what the context reports.
const WASM_THREADS = 1;

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
}: SessionOptions): Promise<EngineSession> {
	// Absolute extension URL — a relative path would resolve against the ort
	// module's own directory and 404 (docs/ENGINE_NOTES.md).
	ort.env.wasm.wasmPaths = wasmBase;
	ort.env.wasm.numThreads = WASM_THREADS;
	ort.env.logLevel = "warning";

	if ((await requestGpuAdapter()) != null) {
		try {
			const session = await ort.InferenceSession.create(modelUrl, {
				executionProviders: ["webgpu"],
			});
			return { backend: "webgpu", session };
		} catch (err) {
			console.warn("[haramblur] WebGPU session failed, falling back to WASM:", err);
		}
	}
	const session = await ort.InferenceSession.create(modelUrl, {
		executionProviders: ["wasm"],
	});
	return { backend: "wasm", session };
}
