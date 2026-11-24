import { vi } from "vitest";
import type {
	ClassifyFrameArgs,
	DocumentLike,
	IntersectionEntryLike,
	IntersectionObserverLike,
	SamplerEnvironment,
} from "../../../src/content/video/types";
import type { Verdict } from "../../../src/shared/protocol";

/** A controllable fake `<video>` element for Node tests. */
export class FakeVideo {
	paused = true;
	ended = false;
	currentTime = 0;
	readyState = 4;
	videoWidth = 640;
	videoHeight = 360;
	dataset: Record<string, string> = {};
	style = {
		filter: "",
		setProperty(k: string, v: string) {
			(this as unknown as Record<string, string>)[k] = v;
		},
	};
	private listeners = new Map<string, Set<() => void>>();

	addEventListener(type: string, fn: () => void) {
		let set = this.listeners.get(type);
		if (!set) {
			set = new Set();
			this.listeners.set(type, set);
		}
		set.add(fn);
	}
	removeEventListener(type: string, fn: () => void) {
		this.listeners.get(type)?.delete(fn);
	}
	dispatch(type: string) {
		for (const fn of [...(this.listeners.get(type) ?? [])]) fn();
	}
	play() {
		this.paused = false;
		this.dispatch("play");
	}
	pause() {
		this.paused = true;
		this.dispatch("pause");
	}
	end() {
		this.ended = true;
		this.dispatch("ended");
	}
}

export function asVideo(v: FakeVideo): HTMLVideoElement {
	return v as unknown as HTMLVideoElement;
}

export class FakeDocument implements DocumentLike {
	hidden = false;
	private listeners = new Set<() => void>();
	addEventListener(_t: "visibilitychange", fn: () => void) {
		this.listeners.add(fn);
	}
	removeEventListener(_t: "visibilitychange", fn: () => void) {
		this.listeners.delete(fn);
	}
	setHidden(hidden: boolean) {
		this.hidden = hidden;
		for (const fn of [...this.listeners]) fn();
	}
}

export class FakeIntersectionObserver implements IntersectionObserverLike {
	private cb: (entries: IntersectionEntryLike[]) => void;
	constructor(cb: (entries: IntersectionEntryLike[]) => void) {
		this.cb = cb;
	}
	observe() {}
	unobserve() {}
	disconnect() {}
	fire(isIntersecting: boolean) {
		this.cb([{ isIntersecting }]);
	}
}

/** Manually pumped frame driver — each `pump()` fires the pending callback. */
export function manualFrameDriver() {
	let pending: (() => void) | null = null;
	return {
		requestFrame: (_video: HTMLVideoElement, cb: () => void) => {
			pending = cb;
			return 1;
		},
		cancelFrame: () => {
			pending = null;
		},
		pump: () => {
			const cb = pending;
			pending = null;
			cb?.();
		},
		hasPending: () => pending !== null,
	};
}

export function verdict(
	decision: "blur" | "clear",
	score = 0.8,
	reasons: number = decision === "blur" ? 1 : 0,
): Verdict {
	return {
		decision,
		reasons: Array.from({ length: reasons }, () => ({
			label: "FACE_FEMALE" as const,
			score,
			box: [0, 0, 1, 1] as [number, number, number, number],
		})),
		ms: 12,
		cached: false,
	};
}

/**
 * Fake canvas: `pixels` is the byte buffer getImageData returns; mutate
 * `canvasSource.nextFill` to change what future canvases contain.
 */
export function fakeCanvasFactory(fill: () => number) {
	return (width: number, height: number) => {
		const data = new Uint8ClampedArray(width * height * 4);
		data.fill(fill());
		return {
			width,
			height,
			getContext: () => ({
				drawImage: () => {},
				getImageData: (_x: number, _y: number, w: number, h: number) => ({
					data: data.subarray(0, w * h * 4),
				}),
			}),
			toDataURL: () => "data:image/png;base64,AAAA",
		} as unknown as import("../../../src/content/video/types").CanvasLike;
	};
}

export interface FakeHarness {
	env: SamplerEnvironment;
	doc: FakeDocument;
	io: FakeIntersectionObserver;
	pumpFrames: () => void;
	now: () => number;
	advance: (ms: number) => void;
	/** Change what future fake canvases contain (simulates new pixels). */
	setFill: (next: () => number) => void;
	classifyFrame: ReturnType<typeof vi.fn<(a: ClassifyFrameArgs) => Promise<Verdict>>>;
}

export function makeHarness(): FakeHarness {
	let now = 1_000_000;
	let fillByte = 0;
	let fillFn: () => number = () => fillByte++ % 256;
	const doc = new FakeDocument();
	const driver = manualFrameDriver();
	let io: FakeIntersectionObserver;
	const timers = new Map<number, { cb: () => void; at: number }>();
	let timerId = 0;
	const env: SamplerEnvironment = {
		now: () => now,
		setTimeout: (cb, ms) => {
			const id = ++timerId;
			timers.set(id, { cb, at: now + ms });
			return id;
		},
		clearTimeout: (id) => timers.delete(id as number),
		requestFrame: driver.requestFrame,
		cancelFrame: driver.cancelFrame,
		document: doc,
		createIntersectionObserver: (cb) => (io = new FakeIntersectionObserver(cb)),
		createCanvas: fakeCanvasFactory(() => fillFn()),
		hostname: () => "example.com",
	};
	return {
		env,
		doc,
		get io() {
			return io;
		},
		pumpFrames: driver.pump,
		now: () => now,
		advance: (ms) => {
			const target = now + ms;
			for (;;) {
				// Fire the earliest due timer; callbacks may schedule more.
				let dueId: number | null = null;
				let dueAt = Infinity;
				for (const [id, t] of timers) {
					if (t.at <= target && t.at < dueAt) {
						dueId = id;
						dueAt = t.at;
					}
				}
				if (dueId === null) break;
				const t = timers.get(dueId);
				timers.delete(dueId);
				now = dueAt;
				t?.cb();
			}
			now = target;
		},
		classifyFrame: vi.fn(async (_a: ClassifyFrameArgs) => verdict("clear")),
		setFill: (next) => {
			fillFn = next;
		},
	};
}
