/**
 * Engine host: owns the worker lifecycle (XState machine), the bounded boot
 * wait, the pixel decode, the verdict cache, and the `policy.decide` wiring
 * (docs/PLAN.md "Engine host"). Runs inside the offscreen document on Chrome
 * and inside the background page on Firefox — identical code both ways.
 */
import { createActor } from "xstate";
import { decide } from "../shared/policy";
import type {
	ClassifyResult,
	Detection,
	EngineRequest,
	EngineStatus,
	Verdict,
} from "../shared/protocol";
import { DEFAULT_SETTINGS, type Settings } from "../shared/settings";
import { LruCache } from "./cache";
import { engineMachine } from "./lifecycle";
import { mergeDetections, tileRects } from "./nudenet";
import type { WorkerInMessage, WorkerOutMessage } from "./worker";

/** Requests during `booting` wait up to this long, then answer `{error}`. */
export const BOOT_TIMEOUT_MS = 30_000;
/** A single inference may not hang forever. */
export const INFER_TIMEOUT_MS = 30_000;
export const VERDICT_CACHE_CAP = 500;

/** The slice of the Worker interface the host needs (injectable for tests). */
export interface WorkerHandle {
	postMessage(message: WorkerInMessage, transfer?: Transferable[]): void;
	terminate(): void;
	onmessage: ((event: MessageEvent<WorkerOutMessage>) => void) | null;
	onerror: ((event: ErrorEvent) => void) | null;
}

export interface EngineHostOptions {
	/** Absolute extension URL of `models/320n.onnx`. */
	modelUrl: string;
	/** Absolute extension URL of the shipped ORT WASM assets (`/ort/`). */
	wasmBase: string;
	spawnWorker?: () => WorkerHandle;
	/** Used only when a request carries no settings (tests). Defaults used otherwise. */
	getSettings?: () => Promise<Settings>;
	/** Pixel decode seam — tests substitute a fake (no ImageBitmap in Node). */
	decodeRequest?: (request: EngineRequest) => Promise<ImageBitmap>;
	bootTimeoutMs?: number;
	inferTimeoutMs?: number;
	cacheCap?: number;
}

export interface EngineHost {
	run(request: EngineRequest): Promise<ClassifyResult>;
	status(): EngineStatus;
	dispose(): void;
}

const defaultSpawn = (): WorkerHandle =>
	new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });

const errMsg = (err: unknown) => (err instanceof Error ? err.message : String(err));

const DATA_URL_RE = /^data:([^;,]*)?(;base64)?,(.*)$/s;

async function decodeDataUrl(dataUrl: string): Promise<ImageBitmap> {
	const m = DATA_URL_RE.exec(dataUrl);
	if (m == null) throw new Error("invalid data URL");
	const mime = m[1] || "image/png";
	const payload = m[3] ?? "";
	const bin = m[2] === ";base64" ? atob(payload) : decodeURIComponent(payload);
	const bytes = new Uint8Array(bin.length);
	for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
	return createImageBitmap(new Blob([bytes], { type: mime }));
}

async function fetchBitmap(url: string): Promise<ImageBitmap> {
	const res = await fetch(url);
	if (!res.ok) throw new Error(`fetch ${url} → HTTP ${res.status}`);
	return createImageBitmap(await res.blob());
}

export function createEngineHost(opts: EngineHostOptions): EngineHost {
	const spawn = opts.spawnWorker ?? defaultSpawn;
	const getS = opts.getSettings ?? (async () => DEFAULT_SETTINGS);
	const bootTimeoutMs = opts.bootTimeoutMs ?? BOOT_TIMEOUT_MS;
	const inferTimeoutMs = opts.inferTimeoutMs ?? INFER_TIMEOUT_MS;
	const cache = new LruCache<string, Verdict>(opts.cacheCap ?? VERDICT_CACHE_CAP);

	const actor = createActor(engineMachine);
	let worker: WorkerHandle | null = null;
	let initTimer: ReturnType<typeof setTimeout> | null = null;
	let inferSeq = 0;
	const pending = new Map<
		number,
		{
			resolve: (detections: Detection[]) => void;
			reject: (err: Error) => void;
			timer: ReturnType<typeof setTimeout>;
		}
	>();

	function failPending(error: string): void {
		for (const p of pending.values()) {
			clearTimeout(p.timer);
			p.reject(new Error(error));
		}
		pending.clear();
	}

	function spawnAndInit(): void {
		worker?.terminate();
		failPending("engine worker restarted");
		worker = spawn();
		worker.onmessage = (event) => handleWorkerMessage(event.data);
		worker.onerror = (event) => crash(event.message || "engine worker error");
		initTimer = setTimeout(
			() => crash(`engine init timed out after ${bootTimeoutMs}ms`),
			bootTimeoutMs,
		);
		worker.postMessage({ type: "init", wasmBase: opts.wasmBase, modelUrl: opts.modelUrl });
	}

	function crash(error: string): void {
		if (initTimer != null) clearTimeout(initTimer);
		initTimer = null;
		failPending(error);
		actor.send({ type: "CRASH", error });
	}

	function handleWorkerMessage(msg: WorkerOutMessage): void {
		switch (msg.type) {
			case "ready":
				if (initTimer != null) clearTimeout(initTimer);
				initTimer = null;
				actor.send({ type: "READY", backend: msg.backend });
				break;
			case "initError":
				crash(msg.error);
				break;
			case "result": {
				const p = pending.get(msg.id);
				if (p == null) break;
				pending.delete(msg.id);
				clearTimeout(p.timer);
				p.resolve(msg.detections);
				break;
			}
			case "inferError": {
				const p = pending.get(msg.id);
				if (p == null) break;
				pending.delete(msg.id);
				clearTimeout(p.timer);
				p.reject(new Error(msg.error));
				break;
			}
		}
	}

	// First spawn is explicit; `context.restarts` ticks up on every
	// CRASH→booting restart, which is what triggers each respawn.
	let lastRestarts = 0;
	actor.start();
	spawnAndInit();
	actor.subscribe((snap) => {
		const restarts = snap.context.restarts;
		if (snap.value === "booting" && restarts !== lastRestarts) {
			lastRestarts = restarts;
			spawnAndInit();
		}
	});

	/**
	 * Resolves when the machine reaches `ready`/`degraded`; returns the error
	 * string on `failed` or when the bounded wait expires (fail-closed path).
	 */
	function waitForReady(timeoutMs: number): Promise<string | null> {
		const snap = actor.getSnapshot();
		if (snap.value === "ready" || snap.value === "degraded") return Promise.resolve(null);
		if (snap.value === "failed") return Promise.resolve(snap.context.error ?? "engine failed");
		return new Promise((resolve) => {
			const timer = setTimeout(() => {
				sub.unsubscribe();
				resolve(`engine boot timed out after ${timeoutMs}ms`);
			}, timeoutMs);
			const sub = actor.subscribe((s) => {
				if (s.value === "ready" || s.value === "degraded") {
					clearTimeout(timer);
					sub.unsubscribe();
					resolve(null);
				} else if (s.value === "failed") {
					clearTimeout(timer);
					sub.unsubscribe();
					resolve(s.context.error ?? "engine failed");
				}
			});
		});
	}

	function infer(bitmap: ImageBitmap, detail: boolean): Promise<Detection[]> {
		const id = ++inferSeq;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				pending.delete(id);
				reject(new Error(`inference timed out after ${inferTimeoutMs}ms`));
			}, inferTimeoutMs);
			pending.set(id, { resolve, reject, timer });
			// No transfer: the bitmap may be needed for a follow-up detail
			// pass; the host closes it once the verdict is decided.
			worker?.postMessage({ type: "infer", id, bitmap, detail });
		});
	}

	const decodeRequest =
		opts.decodeRequest ??
		(async (request: EngineRequest): Promise<ImageBitmap> => {
			if (request.kind === "frame") return decodeDataUrl(request.dataUrl);
			if (request.frameUrl != null) return decodeDataUrl(request.frameUrl);
			return fetchBitmap(request.url);
		});

	async function run(request: EngineRequest): Promise<ClassifyResult> {
		if (request.kind === "image") {
			const hit = cache.get(request.url);
			if (hit != null) return { ...hit, cached: true, ms: 0 };
		}
		const bootError = await waitForReady(bootTimeoutMs);
		if (bootError != null) return { error: bootError };
		const t0 = performance.now();
		try {
			const bitmap = await decodeRequest(request);
			try {
				let detections = await infer(bitmap, false);
				const s = request.settings ?? (await getS());
				// Detail pass (WP7): a `clear` verdict on an image bigger than the
				// model input may hide faces letterboxed below the model floor, so
				// re-scan overlapping crops before committing to clear.
				if (
					decide(detections, s) === "clear" &&
					tileRects(bitmap.width, bitmap.height).length > 0
				) {
					const tiled = await infer(bitmap, true);
					detections = mergeDetections(detections, tiled);
				}
				const verdict: Verdict = {
					decision: decide(detections, s),
					reasons: detections,
					ms: Math.round(performance.now() - t0),
					cached: false,
				};
				if (request.kind === "image") cache.set(request.url, verdict);
				return verdict;
			} finally {
				bitmap.close();
			}
		} catch (err) {
			return { error: errMsg(err) };
		}
	}

	function status(): EngineStatus {
		const snap = actor.getSnapshot();
		return {
			state: snap.value as EngineStatus["state"],
			backend: snap.context.backend,
			error: snap.context.error ?? undefined,
		};
	}

	function dispose(): void {
		if (initTimer != null) clearTimeout(initTimer);
		worker?.terminate();
		failPending("engine host disposed");
		actor.stop();
	}

	return { run, status, dispose };
}
