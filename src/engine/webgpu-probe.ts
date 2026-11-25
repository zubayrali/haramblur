/**
 * DP0 WebGPU reachability probe (docs/DETECTION_PARITY.md). This module is
 * measurement-only: production backend selection stays in `backend.ts`; the
 * probe answers whether the exact extension context can see a WebGPU adapter
 * and what a fresh NudeNet WebGPU session costs on zero tensors.
 */
import type {
	WebGpuContextReport,
	WebGpuProbeContext,
	WebGpuZeroTensorReport,
} from "../shared/webgpu-probe";
import { INPUT_SIZE } from "./nudenet";
import { probeAdapter } from "./webgpu-adapter";

type OrtModule = typeof import("onnxruntime-web");
type OrtSession = Awaited<ReturnType<OrtModule["InferenceSession"]["create"]>>;
type OrtTensor = InstanceType<OrtModule["Tensor"]>;

export interface ProbeWebgpuContextOptions {
	context: WebGpuProbeContext;
	modelUrl: string;
	wasmBase: string;
	/** Number of post-warm-up zero-tensor runs. */
	warmRuns?: number;
	/** Run the ORT WebGPU measurement when an adapter exists. */
	runZeroTensor?: boolean;
	/** Single run timeout; a wedged EP must fail the probe, not hang it. */
	timeoutMs?: number;
}

const DEFAULT_WARM_RUNS = 8;
const DEFAULT_PROBE_TIMEOUT_MS = 15_000;

const errMsg = (err: unknown) => (err instanceof Error ? err.message : String(err));
const roundMs = (ms: number) => Math.round(ms * 100) / 100;
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function emptyZeroTensor(attempted: boolean): WebGpuZeroTensorReport {
	return { attempted, warmRunsMs: [] };
}

function disposeOutputs(outputs: Record<string, OrtTensor> | null | undefined): void {
	if (outputs == null) return;
	for (const tensor of Object.values(outputs)) tensor.dispose();
}

async function runZeroTensorOnce(
	ort: OrtModule,
	session: OrtSession,
	inputSize: number,
	timeoutMs: number,
): Promise<{ ms: number; outputDims: number[] }> {
	const inputName = session.inputNames[0];
	if (inputName == null) throw new Error("model exposes no input");
	const input = new ort.Tensor("float32", new Float32Array(3 * inputSize * inputSize), [
		1,
		3,
		inputSize,
		inputSize,
	]);
	let outputs: Record<string, OrtTensor> | null | undefined;
	const t0 = performance.now();
	try {
		const runPromise = session.run({ [inputName]: input });
		outputs = await Promise.race([runPromise, sleep(timeoutMs).then(() => null)]);
		if (outputs == null) {
			// Avoid an unhandled rejection after the probe has already failed;
			// dispose the late tensors if the wedged run eventually completes.
			void runPromise.then((late) => disposeOutputs(late)).catch(() => undefined);
			throw new Error(`zero-tensor run timed out after ${timeoutMs}ms`);
		}
		const ms = roundMs(performance.now() - t0);
		const first = outputs[session.outputNames[0]!];
		return { ms, outputDims: first == null ? [] : [...first.dims] };
	} finally {
		input.dispose();
		disposeOutputs(outputs);
	}
}

/**
 * Create a fresh WebGPU EP session and time zero-tensor cold/warm inference.
 * A fresh session is deliberate: reusing the production session would hide
 * even per-session cold-start cost. The run can still benefit from browser or
 * driver shader caches warmed by the production boot probe, so `coldRunMs`
 * means first run of this fresh session — not necessarily first-ever compile
 * in the browser process.
 */
async function probeZeroTensor(
	modelUrl: string,
	wasmBase: string,
	warmRuns: number,
	inputSize: number,
	timeoutMs: number,
): Promise<WebGpuZeroTensorReport> {
	const report = emptyZeroTensor(true);
	let session: OrtSession | null = null;
	try {
		const ort = await import("onnxruntime-web");
		ort.env.wasm.wasmPaths = wasmBase;
		ort.env.logLevel = "warning";
		const createStart = performance.now();
		const createPromise = ort.InferenceSession.create(modelUrl, { executionProviders: ["webgpu"] });
		session = await Promise.race([createPromise, sleep(timeoutMs).then(() => null)]);
		if (session == null) {
			// Session creation cannot be cancelled; dispose it if the wedged EP
			// finishes later so the probe does not leak a second NudeNet session.
			void createPromise.then((late) => late.release()).catch(() => undefined);
			throw new Error(`WebGPU session creation timed out after ${timeoutMs}ms`);
		}
		report.sessionCreateMs = roundMs(performance.now() - createStart);

		const cold = await runZeroTensorOnce(ort, session, inputSize, timeoutMs);
		report.coldRunMs = cold.ms;
		report.outputDims = cold.outputDims;
		for (let i = 0; i < warmRuns; i++) {
			const warm = await runZeroTensorOnce(ort, session, inputSize, timeoutMs);
			report.warmRunsMs.push(warm.ms);
		}
		if (report.warmRunsMs.length > 0) {
			report.warmMeanMs = roundMs(
				report.warmRunsMs.reduce((sum, ms) => sum + ms, 0) / report.warmRunsMs.length,
			);
			report.warmMinMs = Math.min(...report.warmRunsMs);
		}
	} catch (err) {
		report.error = errMsg(err);
	} finally {
		if (session != null) await session.release().catch(() => undefined);
	}
	return report;
}

/**
 * Probe one JavaScript context. Adapter state is always collected; the
 * comparatively expensive fresh ORT session runs only when `runZeroTensor` is
 * true and an adapter exists.
 */
export async function probeWebgpuContext({
	context,
	modelUrl,
	wasmBase,
	warmRuns = DEFAULT_WARM_RUNS,
	runZeroTensor = true,
	timeoutMs = DEFAULT_PROBE_TIMEOUT_MS,
}: ProbeWebgpuContextOptions): Promise<WebGpuContextReport> {
	const adapter = await probeAdapter();
	let zeroTensor = emptyZeroTensor(false);
	if (runZeroTensor && adapter.adapter) {
		zeroTensor = await probeZeroTensor(modelUrl, wasmBase, warmRuns, INPUT_SIZE, timeoutMs);
	}
	return { context, adapter, zeroTensor };
}
