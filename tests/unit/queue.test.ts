import { describe, expect, it } from "vitest";
import { TaskQueue } from "../../src/content/queue";

const resolved =
	<T>(v: T) =>
	() =>
		Promise.resolve(v);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe("TaskQueue", () => {
	it("runs visible tasks before background tasks", async () => {
		const q = new TaskQueue({ concurrency: 1 });
		const order: string[] = [];
		const run = (name: string) => () =>
			sleep(2).then(() => {
				order.push(name);
				return name;
			});
		await Promise.all([
			q.enqueue("a", run("a"), "background"),
			q.enqueue("b", run("b"), "background"),
			q.enqueue("c", run("c"), "visible"),
			q.enqueue("d", run("d"), "visible"),
			q.enqueue("e", run("e"), "background"),
		]);
		// "a" was enqueued first and may already be running; every visible task
		// must finish before any remaining background task starts.
		expect(order[0]).toBe("a");
		expect(order.slice(1, 3).sort()).toEqual(["c", "d"]);
		expect(order.slice(3).sort()).toEqual(["b", "e"]);
	});

	it("caps concurrency", async () => {
		const q = new TaskQueue({ concurrency: 4 });
		let running = 0;
		let maxRunning = 0;
		const run = () => async () => {
			running += 1;
			maxRunning = Math.max(maxRunning, running);
			await sleep(5);
			running -= 1;
		};
		await Promise.all(Array.from({ length: 12 }, (_, i) => q.enqueue(`k${i}`, run(), "visible")));
		expect(maxRunning).toBe(4);
	});

	it("dedups by key: one run, shared result", async () => {
		const q = new TaskQueue({ concurrency: 4 });
		let calls = 0;
		const run = () => {
			calls += 1;
			return Promise.resolve(42);
		};
		const [a, b] = await Promise.all([
			q.enqueue("u1", run, "visible"),
			q.enqueue("u1", run, "background"),
		]);
		expect(calls).toBe(1);
		expect(a).toBe(42);
		expect(b).toBe(42);
		// Settled keys stay deduped for the page lifetime.
		expect(await q.enqueue("u1", run)).toBe(42);
		expect(calls).toBe(1);
	});

	it("retries once on transient errors, then rejects", async () => {
		const q = new TaskQueue({ concurrency: 2, retryDelayMs: 1 });
		let attempts = 0;
		const failing = () => {
			attempts += 1;
			return Promise.reject(new Error("transient"));
		};
		await expect(q.enqueue("bad", failing, "visible")).rejects.toThrow("transient");
		expect(attempts).toBe(2); // initial + one retry
		// After rejecting, the key is free again.
		attempts = 0;
		let calls = 0;
		await expect(
			q.enqueue("bad", () => {
				calls += 1;
				return Promise.resolve("ok");
			}),
		).resolves.toBe("ok");
		expect(calls).toBe(1);
	});

	it("succeeds on retry without another enqueue", async () => {
		const q = new TaskQueue({ retryDelayMs: 1 });
		let attempts = 0;
		const flaky = () =>
			++attempts === 1 ? Promise.reject(new Error("boom")) : Promise.resolve("fine");
		await expect(q.enqueue("flaky", flaky)).resolves.toBe("fine");
		expect(attempts).toBe(2);
	});

	it("prioritize() boosts a still-queued task", async () => {
		const q = new TaskQueue({ concurrency: 1 });
		const order: string[] = [];
		const run = (name: string) => () =>
			sleep(2).then(() => {
				order.push(name);
			});
		const first = q.enqueue("a", run("a"), "background");
		const mid = q.enqueue("m", run("m"), "background");
		const late = q.enqueue("z", run("z"), "background");
		q.prioritize("z");
		await Promise.all([first, mid, late]);
		expect(order).toEqual(["a", "z", "m"]);
	});

	it("evicts the oldest settled keys past settledCap; never queued/in-flight", async () => {
		const q = new TaskQueue({ settledCap: 3 });
		for (let i = 0; i < 5; i++) await q.enqueue(`k${i}`, resolved(i));
		// k0 and k1 were evicted — re-enqueueing them runs again.
		let calls = 0;
		const rerun = () => {
			calls += 1;
			return Promise.resolve("rerun");
		};
		expect(await q.enqueue("k0", rerun)).toBe("rerun");
		expect(await q.enqueue("k1", rerun)).toBe("rerun");
		expect(calls).toBe(2);
		// k2..k4 are still deduped.
		expect(await q.enqueue("k4", rerun)).toBe(4);
		expect(calls).toBe(2);
	});

	it("clear() drops settled dedup so keys re-run", async () => {
		const q = new TaskQueue();
		await q.enqueue("u", resolved(1));
		q.clear();
		let calls = 0;
		await q.enqueue("u", () => {
			calls += 1;
			return Promise.resolve(2);
		});
		expect(calls).toBe(1);
	});

	it("enqueue() of a queued duplicate with higher priority boosts it", async () => {
		const q = new TaskQueue({ concurrency: 1 });
		const order: string[] = [];
		const run = (name: string) => () =>
			sleep(2).then(() => {
				order.push(name);
			});
		const a = q.enqueue("a", run("a"), "background");
		const b = q.enqueue("b", run("b"), "background");
		const dupC = q.enqueue("c", run("c"), "background");
		await q.enqueue("c", resolved(undefined), "visible"); // dup: boosts c
		await Promise.all([a, b, dupC]);
		expect(order).toEqual(["a", "c", "b"]);
	});
});
