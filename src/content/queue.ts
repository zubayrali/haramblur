/**
 * Visible-first classification queue (docs/PLAN.md WP2). Shared with WP3: the
 * video sampler enqueues frame classifications through these exports.
 *
 * - `visible` tasks run before `background` ones (FIFO within a priority).
 * - Per-tab concurrency is capped (default 4).
 * - `enqueue` dedups by key: an in-flight or already-settled task for the same
 *   key is returned instead of re-running (the engine has its own LRU verdict
 *   cache across tabs; this is the per-page layer).
 * - Each task retries once on a transient error before rejecting.
 */

export type TaskPriority = "visible" | "background";

export interface TaskQueueOptions {
	/** Max tasks running at once. Default 4. */
	concurrency?: number;
	/** Extra attempts after the first failure. Default 1 (retry once). */
	maxRetries?: number;
	/** Delay before a retry. Default 250 ms. */
	retryDelayMs?: number;
}

interface QueuedTask<T> {
	key: string;
	priority: TaskPriority;
	run: () => Promise<T>;
	resolve: (value: T) => void;
	reject: (reason: unknown) => void;
	attemptsLeft: number;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export class TaskQueue {
	private readonly concurrency: number;
	private readonly maxRetries: number;
	private readonly retryDelayMs: number;
	private inflight = 0;
	private readonly pending: QueuedTask<unknown>[] = [];
	// key → promise for every enqueued task; settled entries stay so re-found
	// media with the same URL never re-requests on this page.
	private readonly byKey = new Map<string, Promise<unknown>>();
	private readonly pendingByKey = new Map<string, QueuedTask<unknown>>();

	constructor(options: TaskQueueOptions = {}) {
		this.concurrency = options.concurrency ?? 4;
		this.maxRetries = options.maxRetries ?? 1;
		this.retryDelayMs = options.retryDelayMs ?? 250;
	}

	/** Tasks currently queued (not running). */
	get pendingCount(): number {
		return this.pending.length;
	}

	/** Tasks currently running. */
	get inflightCount(): number {
		return this.inflight;
	}

	/** Whether a task for `key` exists (queued, running, or settled). */
	has(key: string): boolean {
		return this.byKey.has(key);
	}

	/**
	 * Enqueue `run` under `key`. A second `enqueue` with the same key returns
	 * the existing task's promise — callers sharing one URL share one request.
	 */
	enqueue<T>(
		key: string,
		run: () => Promise<T>,
		priority: TaskPriority = "background",
	): Promise<T> {
		const existing = this.byKey.get(key);
		if (existing != null) {
			// A queued duplicate may still be boosted by a later visible caller.
			const queued = this.pendingByKey.get(key);
			if (queued != null && priority === "visible") queued.priority = "visible";
			return existing as Promise<T>;
		}
		let resolve!: (value: T) => void;
		let reject!: (reason: unknown) => void;
		const promise = new Promise<T>((res, rej) => {
			resolve = res;
			reject = rej;
		});
		const task: QueuedTask<T> = {
			key,
			priority,
			run,
			resolve,
			reject,
			attemptsLeft: this.maxRetries + 1,
		};
		this.pending.push(task as QueuedTask<unknown>);
		this.pendingByKey.set(key, task as QueuedTask<unknown>);
		this.byKey.set(key, promise);
		this.pump();
		return promise;
	}

	/**
	 * Move a still-queued task to the visible priority. Used when an off-screen
	 * element scrolls into view while its classification is still waiting.
	 */
	prioritize(key: string): void {
		const task = this.pendingByKey.get(key);
		if (task != null) task.priority = "visible";
	}

	/**
	 * Drop a settled key so a later `enqueue` re-runs. Rarely needed — verdicts
	 * are stable for a page; kept for WP3's per-frame submissions.
	 */
	delete(key: string): void {
		this.byKey.delete(key);
		this.pendingByKey.delete(key);
	}

	private nextTask(): QueuedTask<unknown> | undefined {
		const visibleIndex = this.pending.findIndex((t) => t.priority === "visible");
		const index = visibleIndex >= 0 ? visibleIndex : this.pending.length > 0 ? 0 : -1;
		if (index < 0) return undefined;
		const task = this.pending.splice(index, 1)[0];
		if (task === undefined) return undefined;
		this.pendingByKey.delete(task.key);
		return task;
	}

	private pump(): void {
		while (this.inflight < this.concurrency) {
			const task = this.nextTask();
			if (task == null) return;
			this.inflight += 1;
			void this.runTask(task).finally(() => {
				this.inflight -= 1;
				this.pump();
			});
		}
	}

	private async runTask<T>(task: QueuedTask<T>): Promise<void> {
		while (task.attemptsLeft > 0) {
			task.attemptsLeft -= 1;
			try {
				const value = await task.run();
				task.resolve(value);
				return;
			} catch (err) {
				if (task.attemptsLeft > 0) await sleep(this.retryDelayMs);
				else {
					this.byKey.delete(task.key);
					task.reject(err);
					return;
				}
			}
		}
	}
}
