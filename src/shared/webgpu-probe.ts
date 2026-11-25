/**
 * Measurement-only WebGPU probe reports (docs/DETECTION_PARITY.md DP0).
 * These types cross the worker/offscreen/background boundary, so they live in
 * shared protocol-adjacent types rather than inside the engine implementation.
 */
export type WebGpuProbeContext = "worker" | "host-main";

export interface WebGpuAdapterInfoReport {
	vendor?: string;
	architecture?: string;
	device?: string;
	description?: string;
}

export interface WebGpuAdapterReport {
	/** `navigator.gpu` existed in this JavaScript context. */
	navigatorGpu: boolean;
	/** `navigator.gpu.requestAdapter()` resolved to a non-null adapter. */
	adapter: boolean;
	/** Adapter request rejection/timeout text, when it failed that way. */
	requestError?: string;
	isFallbackAdapter?: boolean;
	/** WebGPU `shader-f16` feature support on the adapter. */
	shaderF16: boolean;
	features: string[];
	/** Known adapter limits, keyed by their WebGPU spec names. */
	limits: Record<string, number>;
	info?: WebGpuAdapterInfoReport;
}

export interface WebGpuZeroTensorReport {
	/** Whether a dedicated WebGPU EP session/zero-tensor run was attempted. */
	attempted: boolean;
	/** `ort.InferenceSession.create` wall time for the temporary session. */
	sessionCreateMs?: number;
	/** First zero-tensor run, including shader compilation. */
	coldRunMs?: number;
	/** Post-warm-up zero-tensor runs. */
	warmRunsMs: number[];
	warmMeanMs?: number;
	warmMinMs?: number;
	/** Shape of the first model output, copied before tensor disposal. */
	outputDims?: number[];
	error?: string;
}

export interface WebGpuContextReport {
	context: WebGpuProbeContext;
	adapter: WebGpuAdapterReport;
	zeroTensor: WebGpuZeroTensorReport;
}

export interface WebGpuProbeReport {
	contexts: WebGpuContextReport[];
	error?: string;
}
