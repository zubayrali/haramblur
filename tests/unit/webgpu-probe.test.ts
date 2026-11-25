import { describe, expect, it } from "vitest";
import { probeAdapter } from "../../src/engine/webgpu-adapter";

describe("probeAdapter", () => {
	it("reports navigator.gpu absence without throwing", async () => {
		const report = await probeAdapter(null);
		expect(report).toMatchObject({
			navigatorGpu: false,
			adapter: false,
			shaderF16: false,
			features: [],
		});
	});

	it("reports adapter reachability, fp16 support, limits, and info", async () => {
		const report = await probeAdapter({
			requestAdapter: async () => ({
				isFallbackAdapter: false,
				features: new Set(["shader-f16", "texture-compression-bc"]),
				limits: {
					maxBufferSize: 1024,
					maxComputeWorkgroupsPerDimension: 64,
					maxComputeWorkgroupSizeX: 32,
				},
				info: {
					vendor: "apple",
					architecture: "metal",
					device: "m-series",
					description: "test adapter",
				},
			}),
		});
		expect(report.navigatorGpu).toBe(true);
		expect(report.adapter).toBe(true);
		expect(report.shaderF16).toBe(true);
		expect(report.features).toEqual(["shader-f16", "texture-compression-bc"]);
		expect(report.limits).toMatchObject({
			maxBufferSize: 1024,
			maxComputeWorkgroupsPerDimension: 64,
			maxComputeWorkgroupSizeX: 32,
		});
		expect(report.info).toMatchObject({ vendor: "apple", architecture: "metal" });
	});

	it("captures adapter request errors as data", async () => {
		const report = await probeAdapter({
			requestAdapter: async () => {
				throw new Error("denied");
			},
		});
		expect(report.navigatorGpu).toBe(true);
		expect(report.adapter).toBe(false);
		expect(report.requestError).toBe("denied");
	});

	it("bounds a wedged adapter request", async () => {
		const report = await probeAdapter({ requestAdapter: () => new Promise(() => undefined) }, 5);
		expect(report.navigatorGpu).toBe(true);
		expect(report.adapter).toBe(false);
		expect(report.requestError).toMatch(/timed out/);
	});
});
