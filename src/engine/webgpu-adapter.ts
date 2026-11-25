/**
 * Dependency-free WebGPU adapter probe. Kept separate from the ORT timing
 * probe so the background service worker can report adapter reachability
 * without bundling onnxruntime-web.
 */
import type { WebGpuAdapterReport } from "../shared/webgpu-probe";

interface GpuFeatureSetLike {
	has?(feature: string): boolean;
	[Symbol.iterator]?: () => Iterator<string>;
}

interface GpuAdapterLike {
	isFallbackAdapter?: boolean;
	features?: GpuFeatureSetLike;
	limits?: Record<string, number>;
	info?: {
		vendor?: unknown;
		architecture?: unknown;
		device?: unknown;
		description?: unknown;
	};
}

export interface GpuLike {
	requestAdapter(): Promise<GpuAdapterLike | null>;
}

// Keys copied explicitly because GPUSupportedLimits is not reliably
// enumerable across implementations. They are the DP0 decision inputs plus
// the texture/storage limits that matter for later input-size work.
const LIMIT_KEYS = [
	"maxBufferSize",
	"maxStorageBufferBindingSize",
	"maxUniformBufferBindingSize",
	"maxComputeWorkgroupSizeX",
	"maxComputeWorkgroupSizeY",
	"maxComputeWorkgroupSizeZ",
	"maxComputeInvocationsPerWorkgroup",
	"maxComputeWorkgroupsPerDimension",
	"maxTextureDimension1D",
	"maxTextureDimension2D",
	"maxTextureDimension3D",
	"maxTextureArrayLayers",
] as const;

const errMsg = (err: unknown) => (err instanceof Error ? err.message : String(err));
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const ADAPTER_TIMEOUT: unique symbol = Symbol("adapter-timeout");
const DEFAULT_ADAPTER_TIMEOUT_MS = 5_000;

function defaultGpu(): GpuLike | null {
	if (typeof navigator === "undefined") return null;
	return (navigator as { gpu?: GpuLike }).gpu ?? null;
}

function adapterFeatures(features: GpuFeatureSetLike | undefined): string[] {
	if (features?.[Symbol.iterator] == null) return [];
	try {
		return Array.from(features as Iterable<string>);
	} catch {
		return [];
	}
}

function adapterLimits(limits: Record<string, number> | undefined): Record<string, number> {
	const out: Record<string, number> = {};
	if (limits == null) return out;
	for (const key of LIMIT_KEYS) {
		const value = limits[key];
		if (typeof value === "number" && Number.isFinite(value)) out[key] = value;
	}
	return out;
}

/** Probe only the WebGPU adapter surface; injectable for unit tests. */
export async function probeAdapter(
	gpu: GpuLike | null = defaultGpu(),
	timeoutMs = DEFAULT_ADAPTER_TIMEOUT_MS,
): Promise<WebGpuAdapterReport> {
	if (gpu == null) {
		return { navigatorGpu: false, adapter: false, shaderF16: false, features: [], limits: {} };
	}
	let adapter: GpuAdapterLike | null = null;
	let requestError: string | undefined;
	try {
		const result: GpuAdapterLike | null | typeof ADAPTER_TIMEOUT = await Promise.race([
			gpu.requestAdapter(),
			sleep(timeoutMs).then((): typeof ADAPTER_TIMEOUT => ADAPTER_TIMEOUT),
		]);
		if (result === ADAPTER_TIMEOUT) {
			requestError = `adapter request timed out after ${timeoutMs}ms`;
		} else {
			adapter = result;
		}
	} catch (err) {
		requestError = errMsg(err);
	}
	const features = adapterFeatures(adapter?.features);
	const shaderF16 =
		(adapter?.features?.has?.("shader-f16") ?? false) || features.includes("shader-f16");
	const info = adapter?.info;
	return {
		navigatorGpu: true,
		adapter: adapter != null,
		...(requestError != null ? { requestError } : {}),
		...(typeof adapter?.isFallbackAdapter === "boolean"
			? { isFallbackAdapter: adapter.isFallbackAdapter }
			: {}),
		shaderF16,
		features,
		limits: adapterLimits(adapter?.limits),
		...(info != null
			? {
					info: {
						...(typeof info.vendor === "string" ? { vendor: info.vendor } : {}),
						...(typeof info.architecture === "string" ? { architecture: info.architecture } : {}),
						...(typeof info.device === "string" ? { device: info.device } : {}),
						...(typeof info.description === "string" ? { description: info.description } : {}),
					},
				}
			: {}),
	};
}
