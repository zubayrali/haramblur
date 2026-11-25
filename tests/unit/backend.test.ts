import { beforeEach, describe, expect, it, vi } from "vitest";

const { createMock } = vi.hoisted(() => ({ createMock: vi.fn() }));

vi.mock("onnxruntime-web", () => ({
	env: { wasm: {}, logLevel: "warning" },
	Tensor: class {
		constructor(
			public type: string,
			public data: unknown,
			public dims: number[],
		) {}
		dispose(): void {}
	},
	InferenceSession: { create: createMock },
}));

import { adapterSupportsShaderF16, createSession } from "../../src/engine/backend";

function fakeSession(runImpl?: () => Promise<unknown>) {
	return {
		inputNames: ["input"],
		outputNames: ["out"],
		inputMetadata: [{ isTensor: true, type: "float32", shape: [1, 3, 4, 4] }],
		run: vi.fn(runImpl ?? (async () => ({ out: { dispose: vi.fn(), dims: [1] } }))),
		release: vi.fn(async () => undefined),
	};
}

const urls = {
	wasmBase: "chrome-extension://x/ort/",
	modelUrl: "chrome-extension://x/models/320n.onnx",
	faceModelUrl: "chrome-extension://x/models/fastface-small-112.onnx",
	yunetModelUrl: "chrome-extension://x/models/yunet-2023mar.onnx",
	nsfwModelUrl: "chrome-extension://x/models/nsfwjs-u8.onnx",
	personModelUrl: "chrome-extension://x/models/yolov8n-seg-fp16.onnx",
	personModelFp32Url: "chrome-extension://x/models/yolov8n-seg-fp32.onnx",
};

function stubGpu(shaderF16: boolean): void {
	vi.stubGlobal("navigator", {
		gpu: {
			requestAdapter: async () => ({
				features: { has: (f: string) => f === "shader-f16" && shaderF16 },
			}),
		},
	});
}

describe("createSession backend selection", () => {
	beforeEach(() => {
		createMock.mockReset();
		vi.unstubAllGlobals();
	});

	it("uses WebGPU independently for every model when warm-up succeeds", async () => {
		stubGpu(true);
		createMock.mockImplementation(async () => fakeSession());
		const loaded = await createSession(urls);
		expect(loaded.backends).toEqual({
			nudenet: "webgpu",
			fastface: "webgpu",
			yunet: "webgpu",
			nsfw: "webgpu",
			person: "webgpu",
		});
		expect(createMock).toHaveBeenCalledTimes(5);
		expect(createMock.mock.calls.every((call) => call[1].executionProviders[0] === "webgpu")).toBe(
			true,
		);
	});

	it("falls back only the failing model to WASM", async () => {
		stubGpu(true);
		createMock.mockImplementation(async (src: string, opts: { executionProviders: string[] }) =>
			fakeSession(async () => {
				if (src.includes("320n") && opts.executionProviders[0] === "webgpu")
					throw new Error("wedged");
				return { out: { dispose: vi.fn(), dims: [1] } };
			}),
		);
		const loaded = await createSession(urls);
		expect(loaded.backend).toBe("wasm");
		expect(loaded.backends.nudenet).toBe("wasm");
		expect(loaded.backends.fastface).toBe("webgpu");
		const nudenetCalls = createMock.mock.calls.filter((call) => call[0].includes("320n"));
		expect(nudenetCalls.map((call) => call[1].executionProviders[0])).toEqual(["webgpu", "wasm"]);
	});

	it("rejects a WebGPU adapter whose steady-state probe exceeds the budget", async () => {
		stubGpu(true);
		createMock.mockImplementation(async () => fakeSession());
		// Fake sessions resolve instantly, so simulate a software adapter by
		// advancing the clock 200ms during the timed (second) probe run.
		const realNow = performance.now.bind(performance);
		let calls = 0;
		vi.spyOn(performance, "now").mockImplementation(() => {
			calls++;
			// Every probe pair has two clock reads; make the measured gap 200ms.
			return realNow() + (calls % 2 === 0 ? 200 : 0);
		});
		const loaded = await createSession(urls);
		expect(loaded.backends.nudenet).toBe("wasm");
		expect(loaded.backends.fastface).toBe("wasm");
		vi.restoreAllMocks();
	});

	it("selects the fp32 person pack when shader-f16 is absent", async () => {
		stubGpu(false);
		createMock.mockImplementation(async () => fakeSession());
		const loaded = await createSession(urls);
		expect(loaded.backends.person).toBe("webgpu");
		const personCalls = createMock.mock.calls.filter((call) => call[0].includes("yolov8n-seg"));
		expect(personCalls).toHaveLength(1);
		expect(personCalls[0]![0]).toContain("fp32");
	});

	it("forceWasm bypasses WebGPU for every model", async () => {
		stubGpu(true);
		createMock.mockImplementation(async () => fakeSession());
		const loaded = await createSession({ ...urls, forceWasm: true });
		expect(loaded.backends).toEqual({
			nudenet: "wasm",
			fastface: "wasm",
			yunet: "wasm",
			nsfw: "wasm",
			person: "wasm",
		});
		expect(createMock.mock.calls.every((call) => call[1].executionProviders[0] === "wasm")).toBe(
			true,
		);
	});

	it("detects shader-f16 support through the adapter feature set", () => {
		expect(adapterSupportsShaderF16({ features: { has: (f) => f === "shader-f16" } })).toBe(true);
		expect(adapterSupportsShaderF16({ features: { has: () => false } })).toBe(false);
		expect(adapterSupportsShaderF16(null)).toBe(false);
	});
});
