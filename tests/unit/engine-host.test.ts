import { describe, expect, it } from "vitest";
import { createEngineHost, type WorkerHandle } from "../../src/engine/host";
import type { WorkerInMessage, WorkerOutMessage } from "../../src/engine/worker";
import type { Detection, EngineBackend, EngineRequest } from "../../src/shared/protocol";
import { DEFAULT_SETTINGS } from "../../src/shared/settings";

const FAKE_BITMAP = { width: 8, height: 8, close: () => {} } as unknown as ImageBitmap;
const BIG_BITMAP = { width: 960, height: 640, close: () => {} } as unknown as ImageBitmap;
const det = (label: Detection["label"], score: number): Detection => ({
	label,
	score,
	box: [0.1, 0.1, 0.2, 0.2],
});

class FakeWorker implements WorkerHandle {
	onmessage: ((event: MessageEvent<WorkerOutMessage>) => void) | null = null;
	onerror: ((event: ErrorEvent) => void) | null = null;
	posted: WorkerInMessage[] = [];
	constructor(
		private opts: {
			backend?: EngineBackend;
			failInit?: boolean;
			silent?: boolean;
			detections?: Detection[];
			detailDetections?: Detection[];
			inferError?: string;
		} = {},
	) {}
	postMessage(msg: WorkerInMessage): void {
		this.posted.push(msg);
		if (this.opts.silent) return;
		queueMicrotask(() => {
			if (msg.type === "init") {
				if (this.opts.failInit) {
					this.onmessage?.({ data: { type: "initError", error: "init boom" } } as MessageEvent);
				} else {
					this.onmessage?.({
						data: { type: "ready", backend: this.opts.backend ?? "wasm" },
					} as MessageEvent);
				}
			} else if (msg.type === "infer") {
				if (this.opts.inferError) {
					this.onmessage?.({
						data: { type: "inferError", id: msg.id, error: this.opts.inferError },
					} as MessageEvent);
				} else {
					this.onmessage?.({
						data: {
							type: "result",
							id: msg.id,
							detections: msg.detail
								? (this.opts.detailDetections ?? this.opts.detections ?? [])
								: (this.opts.detections ?? []),
							ms: 1,
						},
					} as MessageEvent);
				}
			}
		});
	}
	terminate(): void {}
}

function makeHost(
	workerOpts: ConstructorParameters<typeof FakeWorker>[0] = {},
	bitmap: ImageBitmap = FAKE_BITMAP,
) {
	const workers: FakeWorker[] = [];
	const host = createEngineHost({
		modelUrl: "chrome-extension://x/models/320n.onnx",
		wasmBase: "chrome-extension://x/ort/",
		spawnWorker: () => {
			const w = new FakeWorker(workerOpts);
			workers.push(w);
			return w;
		},
		getSettings: async () => DEFAULT_SETTINGS,
		decodeRequest: async () => bitmap,
	});
	return { host, workers };
}

const imageReq: EngineRequest = { kind: "image", url: "https://x/img.jpg", width: 8, height: 8 };

describe("engine host", () => {
	it("reports booting → degraded on WASM init", async () => {
		const { host } = makeHost({ backend: "wasm" });
		const verdict = await host.run(imageReq);
		expect(host.status().state).toBe("degraded");
		expect(host.status().backend).toBe("wasm");
		expect("error" in verdict ? verdict.error : verdict.decision).toBe("clear");
		host.dispose();
	});

	it("applies policy.decide: female face blurs, gradient clears", async () => {
		const { host } = makeHost({ detections: [det("FACE_FEMALE", 0.9)] });
		const res = await host.run(imageReq);
		expect(res).toMatchObject({ decision: "blur", cached: false });
		expect((res as { reasons: Detection[] }).reasons[0]!.label).toBe("FACE_FEMALE");
		host.dispose();
	});

	it("caches image verdicts by URL and never caches frames", async () => {
		const { host, workers } = makeHost({ detections: [det("FACE_FEMALE", 0.9)] });
		const first = await host.run(imageReq);
		const second = await host.run(imageReq);
		expect(first).toMatchObject({ decision: "blur", cached: false });
		expect(second).toMatchObject({ decision: "blur", cached: true });
		const infers = workers[0]!.posted.filter((m) => m.type === "infer");
		expect(infers).toHaveLength(1);
		const frame = await host.run({
			kind: "frame",
			dataUrl: "data:image/png;base64,",
			videoId: "v",
			ts: 0,
		});
		expect(frame).toMatchObject({ cached: false });
		host.dispose();
	});

	it("runs the detail pass only when pass one is clear on a large image", async () => {
		// Pass 1 sees nothing; tiles find a female face → blur via 2nd infer.
		const { host, workers } = makeHost(
			{ detections: [], detailDetections: [det("FACE_FEMALE", 0.9)] },
			BIG_BITMAP,
		);
		const res = await host.run({ ...imageReq, width: 960, height: 640 });
		expect(res).toMatchObject({ decision: "blur" });
		const infers = workers[0]!.posted.filter(
			(m): m is Extract<WorkerInMessage, { type: "infer" }> => m.type === "infer",
		);
		expect(infers.map((m) => m.detail === true)).toEqual([false, true]);
		host.dispose();
	});

	it("skips the detail pass on a blur hit or a small image", async () => {
		// Pass 1 already blurs → single infer even on a big source.
		const { host, workers } = makeHost({ detections: [det("FACE_FEMALE", 0.9)] }, BIG_BITMAP);
		await host.run({ ...imageReq, width: 960, height: 640 });
		expect(workers[0]!.posted.filter((m) => m.type === "infer")).toHaveLength(1);
		host.dispose();

		// Clear verdict but source fits the input → no tiles, single infer.
		const h2 = makeHost({ detections: [] });
		const res2 = await h2.host.run(imageReq);
		expect(res2).toMatchObject({ decision: "clear" });
		expect(h2.workers[0]!.posted.filter((m) => m.type === "infer")).toHaveLength(1);
		h2.host.dispose();
	});

	it("returns {error} when boot exceeds the bounded wait", async () => {
		const { host } = makeHost({ silent: true });
		host.dispose();
		const h2 = createEngineHost({
			modelUrl: "x",
			wasmBase: "y",
			spawnWorker: () => new FakeWorker({ silent: true }),
			getSettings: async () => DEFAULT_SETTINGS,
			decodeRequest: async () => FAKE_BITMAP,
			bootTimeoutMs: 60,
		});
		const res = await h2.run(imageReq);
		expect(res).toHaveProperty("error");
		expect((res as { error: string }).error).toMatch(/timed out|failed/);
		h2.dispose();
	});

	it("fails once-restarted then stays failed", async () => {
		const { host, workers } = makeHost({ failInit: true });
		const res = await host.run({ kind: "frame", dataUrl: "data:,", videoId: "v", ts: 0 });
		expect(res).toHaveProperty("error");
		// First initError → restart → second initError → failed.
		expect(workers).toHaveLength(2);
		expect(host.status().state).toBe("failed");
		host.dispose();
	});

	it("propagates inference errors as {error}", async () => {
		const { host } = makeHost({ inferError: "oom" });
		const res = await host.run({ kind: "frame", dataUrl: "data:,", videoId: "v", ts: 0 });
		expect(res).toEqual({ error: "oom" });
		host.dispose();
	});
});
