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
import type {
	WebGpuContextReport,
	WebGpuProbeReport,
	WebGpuZeroTensorReport,
} from "../shared/webgpu-probe";
import { LruCache } from "./cache";
import { boxIou } from "./fastface";
import { engineMachine } from "./lifecycle";
import { mergeDetections, tileRects } from "./nudenet";
import { probeAdapter } from "./webgpu-adapter";
import type { WorkerInMessage, WorkerOutMessage } from "./worker";

/** Requests during `booting` wait up to this long, then answer `{error}`. */
export const BOOT_TIMEOUT_MS = 30_000;
/** A single inference may not hang forever. */
export const INFER_TIMEOUT_MS = 30_000;
/** DP0 includes shader compilation, so it shares the bounded inference budget. */
export const WEBGPU_PROBE_TIMEOUT_MS = 30_000;
/** Warm runs after the shader-compiling cold run. */
export const WEBGPU_PROBE_WARM_RUNS = 8;
/** Abort a stalled image fetch so it can't hold a queue slot forever. */
export const FETCH_TIMEOUT_MS = 15_000;
export const VERDICT_CACHE_CAP = 500;
/** Minimum time between person segmentation passes on one media element.
 * Bounded worst-case cost: even at the 10 fps continuous sampler ceiling at
 * most one frame in nine carries a pass, and only when the faces actually moved. */
export const PERSON_PASS_INTERVAL_MS = 1500;

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
	/** Absolute extension URL of `models/fastface-small-112.onnx` (optional). */
	faceModelUrl?: string;
	/** Absolute extension URL of `models/yunet-2023mar.onnx` (optional). */
	yunetModelUrl?: string;
	/** Absolute extension URL of `models/nsfwjs-u8.onnx` (optional). */
	nsfwModelUrl?: string;
	/** Absolute extension URL of `models/yolov8n-seg-fp16.onnx` (optional). */
	personModelUrl?: string;
	/** FP32 person-segmentation fallback when adapter shader-f16 is absent. */
	personModelFp32Url?: string;
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
	/** Minimum milliseconds between person segmentation passes per media key. */
	personPassIntervalMs?: number;
	/** Clock seam for tests (performance.now default). */
	now?: () => number;
}

export interface EngineHost {
	run(request: EngineRequest): Promise<ClassifyResult>;
	status(): EngineStatus;
	/** DP0 measurement probe; measurement-only, never changes backend selection. */
	probeWebgpu(): Promise<WebGpuProbeReport>;
	dispose(): void;
}

const defaultSpawn = (): WorkerHandle =>
	new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });

const errMsg = (err: unknown) => (err instanceof Error ? err.message : String(err));
const emptyZeroTensor = (): WebGpuZeroTensorReport => ({ attempted: false, warmRunsMs: [] });

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
	const res = await fetch(url, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) });
	if (!res.ok) throw new Error(`fetch ${url} → HTTP ${res.status}`);
	return createImageBitmap(await res.blob());
}

/**
 * Minimum face-box IoU for a prior frame's silhouette mask to carry onto the
 * current frame's matching FACE_FEMALE detection. Faces move little between
 * frames at the continuous sampling cadence; faster motion changes the quantized
 * female-verdict key, which grants a fresh segmentation pass anyway.
 */
export const MASK_CARRY_MIN_IOU = 0.2;

/**
 * Copy silhouette masks from the previous frame's detections onto the current
 * frame's matching female faces, in place. Person passes are intentionally
 * rare (one per verdict-key transition), but region rendering happens on every
 * verdict — without carry-over the silhouette flashes for a single frame and
 * the censor reverts to the head-ratio heuristic box on every other frame.
 * Masks are immutable wire data, so sharing the reference is safe.
 */
export function carryMasksForward(prior: readonly Detection[], current: Detection[]): void {
	const maskedFaces = prior.filter((d) => d.label === "FACE_FEMALE" && d.mask != null);
	if (maskedFaces.length === 0) return;
	for (const det of current) {
		if (det.label !== "FACE_FEMALE" || det.mask != null) continue;
		let best: Detection | null = null;
		let bestIou = MASK_CARRY_MIN_IOU;
		for (const candidate of maskedFaces) {
			const iou = boxIou(det.box, candidate.box);
			if (iou > bestIou) {
				bestIou = iou;
				best = candidate;
			}
		}
		if (best?.mask != null) det.mask = best.mask;
	}
}

export function createEngineHost(opts: EngineHostOptions): EngineHost {
	const spawn = opts.spawnWorker ?? defaultSpawn;
	const getS = opts.getSettings ?? (async () => DEFAULT_SETTINGS);
	const bootTimeoutMs = opts.bootTimeoutMs ?? BOOT_TIMEOUT_MS;
	const inferTimeoutMs = opts.inferTimeoutMs ?? INFER_TIMEOUT_MS;
	const cacheCap = opts.cacheCap ?? VERDICT_CACHE_CAP;
	const personPassIntervalMs = opts.personPassIntervalMs ?? PERSON_PASS_INTERVAL_MS;
	const now = opts.now ?? (() => performance.now());
	const cache = new LruCache<string, { detections: Detection[]; detailed: boolean }>(cacheCap);
	/** Latest detections per video/image key, including videos that have no URL cache. */
	const lastDetections = new LruCache<string, Detection[]>(cacheCap);
	/** Last granted person pass per media key: when it ran and the female-face
	 * boxes of the frame it ran on (movement baseline for the next grant). */
	const personPassState = new LruCache<string, { at: number; boxes: Detection["box"][] }>(cacheCap);

	const actor = createActor(engineMachine);
	let worker: WorkerHandle | null = null;
	let initTimer: ReturnType<typeof setTimeout> | null = null;
	let inferSeq = 0;
	let probeSeq = 0;
	const pending = new Map<
		number,
		{
			resolve: (detections: Detection[]) => void;
			reject: (err: Error) => void;
			timer: ReturnType<typeof setTimeout>;
		}
	>();
	const probePending = new Map<
		number,
		{
			resolve: (report: WebGpuContextReport) => void;
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
		for (const p of probePending.values()) {
			clearTimeout(p.timer);
			p.reject(new Error(error));
		}
		probePending.clear();
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
		worker.postMessage({
			type: "init",
			wasmBase: opts.wasmBase,
			modelUrl: opts.modelUrl,
			faceModelUrl: opts.faceModelUrl,
			yunetModelUrl: opts.yunetModelUrl,
			nsfwModelUrl: opts.nsfwModelUrl,
			personModelUrl: opts.personModelUrl,
			personModelFp32Url: opts.personModelFp32Url,
		});
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
				actor.send({ type: "READY", backend: msg.backend, backends: msg.backends });
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
			case "webgpuProbeResult": {
				const p = probePending.get(msg.id);
				if (p == null) break;
				probePending.delete(msg.id);
				clearTimeout(p.timer);
				p.resolve(msg.report);
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

	function infer(
		bitmap: ImageBitmap,
		detail: boolean,
		settings: Settings,
		person = false,
	): Promise<Detection[]> {
		const id = ++inferSeq;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				pending.delete(id);
				reject(new Error(`inference timed out after ${inferTimeoutMs}ms`));
			}, inferTimeoutMs);
			pending.set(id, { resolve, reject, timer });
			// No transfer: the bitmap may be needed for a follow-up detail
			// pass; the host closes it once the verdict is decided.
			worker?.postMessage({ type: "infer", id, bitmap, detail, settings, person });
		});
	}

	function requestWorkerProbe(): Promise<WebGpuContextReport> {
		const id = ++probeSeq;
		return new Promise((resolve, reject) => {
			if (worker == null) {
				reject(new Error("engine worker unavailable"));
				return;
			}
			const timer = setTimeout(() => {
				probePending.delete(id);
				reject(new Error(`WebGPU probe timed out after ${WEBGPU_PROBE_TIMEOUT_MS}ms`));
			}, WEBGPU_PROBE_TIMEOUT_MS);
			probePending.set(id, { resolve, reject, timer });
			worker.postMessage({
				type: "probeWebgpu",
				id,
				modelUrl: opts.modelUrl,
				wasmBase: opts.wasmBase,
				warmRuns: WEBGPU_PROBE_WARM_RUNS,
			});
		});
	}

	const decodeRequest =
		opts.decodeRequest ??
		(async (request: EngineRequest): Promise<ImageBitmap> => {
			if (request.kind === "frame") return decodeDataUrl(request.dataUrl);
			if (request.frameUrl != null) return decodeDataUrl(request.frameUrl);
			return fetchBitmap(request.url);
		});

	/**
	 * Female faces of a blur verdict, sorted for stable comparison. Person
	 * passes only make sense when the verdict actually carries a female face —
	 * the silhouette association target.
	 */
	function femaleBlurFaces(
		detections: readonly Detection[] | undefined,
		settings: Settings,
	): Detection["box"][] {
		if (detections == null || decide(detections, settings) !== "blur") return [];
		return detections
			.filter((d) => d.label === "FACE_FEMALE")
			.map((d) => d.box)
			.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
	}

	/**
	 * Whether two sorted face-box sets differ beyond detector jitter: a face
	 * count change, or any face whose center or size moved more than 5% of the
	 * frame. Distance-based comparison avoids the boundary flip-flopping of the
	 * earlier quantized keys, which granted a segmentation pass every other
	 * frame whenever a face sat near a quantization edge.
	 */
	function facesMoved(a: Detection["box"][], b: Detection["box"][]): boolean {
		if (a.length !== b.length) return true;
		for (let i = 0; i < a.length; i++) {
			const [ax, ay, aw, ah] = a[i]!;
			const [bx, by, bw, bh] = b[i]!;
			if (
				Math.abs(ax + aw / 2 - (bx + bw / 2)) > 0.05 ||
				Math.abs(ay + ah / 2 - (by + bh / 2)) > 0.05 ||
				Math.abs(aw - bw) > 0.05 ||
				Math.abs(ah - bh) > 0.05
			) {
				return true;
			}
		}
		return false;
	}

	async function run(request: EngineRequest): Promise<ClassifyResult> {
		const s = request.settings ?? (await getS());
		// The cache stores raw detections, not verdicts: `decide()` is cheap and
		// must re-run against current settings so strictness/gender toggles take
		// effect on cached URLs. `detailed` records whether the tiled detail
		// pass already ran — a cached non-detailed `clear` still needs it.
		const mediaKey = request.kind === "image" ? `image:${request.url}` : `frame:${request.videoId}`;
		// Silhouettes are only paintable in regional body-coverage mode; in face/
		// head mode or whole-image mode the pass would be pure overhead. The image
		// cache key carries that eligibility so turning body coverage on later
		// re-runs inference instead of reusing maskless detections.
		const personUseful =
			s.coverageMode === "body" && s.regionBlur && !(s.wholeImages && request.kind === "image");
		const cacheKey =
			request.kind === "image" ? `${request.url}|person:${personUseful ? 1 : 0}` : "";
		const hit = request.kind === "image" ? cache.get(cacheKey) : undefined;
		const priorDetections = hit?.detections ?? lastDetections.get(mediaKey);
		const priorFemaleFaces = femaleBlurFaces(priorDetections, s);
		const passState = personPassState.get(mediaKey);
		if (hit != null) {
			const decision = decide(hit.detections, s);
			if (decision === "blur" || hit.detailed) {
				return { decision, reasons: hit.detections, ms: 0, cached: true };
			}
		}
		const bootError = await waitForReady(bootTimeoutMs);
		if (bootError != null) return { error: bootError };
		const t0 = performance.now();
		try {
			const bitmap = await decodeRequest(request);
			try {
				let detections: Detection[];
				let detailed = false;
				let personAttempted = false;
				const tiles = tileRects(bitmap.width, bitmap.height);
				if (hit != null) {
					// Cached non-detailed clear: skip the whole pass, run detail only.
					detections = [...hit.detections];
					if (tiles.length > 0) {
						const tiled = await infer(bitmap, true, s, false);
						detections = mergeDetections(detections, tiled);
						detailed = true;
					}
				} else {
					// Person passes are expensive, so grants are rare: once per media
					// element initially, then only when the female faces actually moved
					// (new scene, subject motion) and never more often than the rate
					// cap. Carry-forward covers the frames in between. Hysteresis
					// delays face-carried video blurs by a frame anyway, so waiting for
					// the first blur verdict costs nothing visually on video.
					const firstPersonGrant =
						personUseful && request.kind === "image" && priorDetections == null;
					const intervalElapsed = passState == null || now() - passState.at >= personPassIntervalMs;
					const transitionPersonGrant =
						personUseful &&
						priorFemaleFaces.length > 0 &&
						intervalElapsed &&
						(passState == null || facesMoved(priorFemaleFaces, passState.boxes));
					const personGrant = firstPersonGrant || transitionPersonGrant;
					personAttempted = personGrant;
					detections = await infer(bitmap, false, s, personGrant);
					// Detail pass (WP7): a `clear` verdict on an image bigger than the
					// model input may hide faces letterboxed below the model floor, so
					// re-scan overlapping crops before committing to clear.
					if (decide(detections, s) === "clear" && tiles.length > 0) {
						// No person flag on detail passes: silhouette segmentation is
						// owned by the whole-frame grant path (rate-capped, movement-
						// triggered). Faces small enough to need tiles are too small
						// for silhouette refinement to matter, and unconditional
						// person flags here would bypass the cap on every clear frame.
						const tiled = await infer(bitmap, true, s, false);
						detections = mergeDetections(detections, tiled);
						detailed = true;
					}
				}
				if (personAttempted) {
					personPassState.set(mediaKey, { at: now(), boxes: femaleBlurFaces(detections, s) });
				} else if (
					request.kind === "frame" &&
					personUseful &&
					priorDetections != null &&
					!detections.some((d) => d.mask != null)
				) {
					carryMasksForward(priorDetections, detections);
				}
				const verdict: Verdict = {
					decision: decide(detections, s),
					reasons: detections,
					ms: Math.round(performance.now() - t0),
					cached: false,
				};
				lastDetections.set(mediaKey, detections);
				if (request.kind === "image") {
					cache.set(cacheKey, { detections, detailed });
				}
				return verdict;
			} finally {
				bitmap.close();
			}
		} catch (err) {
			return { error: errMsg(err) };
		}
	}

	async function probeWebgpu(): Promise<WebGpuProbeReport> {
		// Always collect the host-main adapter surface. If the dedicated worker
		// cannot see/use WebGPU, DP0 needs this comparison to decide whether the
		// engine host architecture — not the GPU itself — is the blocker.
		const hostAdapter = await probeAdapter();
		const hostMainSkipped: WebGpuContextReport = {
			context: "host-main",
			adapter: hostAdapter,
			zeroTensor: emptyZeroTensor(),
		};
		const bootError = await waitForReady(bootTimeoutMs);
		if (bootError != null) return { contexts: [hostMainSkipped], error: bootError };

		let workerReport: WebGpuContextReport;
		let probeTransportError: string | undefined;
		try {
			workerReport = await requestWorkerProbe();
		} catch (err) {
			probeTransportError = errMsg(err);
			workerReport = {
				context: "worker",
				adapter: {
					navigatorGpu: false,
					adapter: false,
					shaderF16: false,
					features: [],
					limits: {},
				},
				zeroTensor: { attempted: true, warmRunsMs: [], error: probeTransportError },
			};
		}

		return {
			contexts: [workerReport, hostMainSkipped],
			...(probeTransportError != null ? { error: probeTransportError } : {}),
		};
	}

	function status(): EngineStatus {
		const snap = actor.getSnapshot();
		return {
			state: snap.value as EngineStatus["state"],
			backend: snap.context.backend,
			...(snap.context.backends != null ? { backends: snap.context.backends } : {}),
			error: snap.context.error ?? undefined,
		};
	}

	function dispose(): void {
		if (initTimer != null) clearTimeout(initTimer);
		worker?.terminate();
		failPending("engine host disposed");
		actor.stop();
	}

	return { run, status, probeWebgpu, dispose };
}
