import { type EngineStatus, sendMessage, type Verdict } from "../shared/protocol";

/**
 * Content-side client for the ML engine (docs/PLAN.md "Message protocol").
 * Calls go through `sendMessage`; the background routes them to the engine
 * host (offscreen document on Chrome, background page on Firefox).
 *
 * While the engine is booting, requests wait — bounded, default 30 s — then
 * fail. The content pipeline applies `settings.failClosed` to that failure.
 */

/** How long a request may wait for the engine to finish booting. */
export const ENGINE_WAIT_MS = 30_000;
const ENGINE_POLL_MS = 400;

export class EngineUnavailableError extends Error {
	readonly status: EngineStatus;
	constructor(status: EngineStatus) {
		super(`engine unavailable: ${status.state}${status.error ? ` (${status.error})` : ""}`);
		this.name = "EngineUnavailableError";
		this.status = status;
	}
}

export function engineStatus(): Promise<EngineStatus> {
	return sendMessage("engineStatus");
}

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

async function pollEngine(timeoutMs: number): Promise<EngineStatus> {
	const deadline = Date.now() + timeoutMs;
	let lastError = "no response from background";
	while (Date.now() < deadline) {
		try {
			const status = await engineStatus();
			if (status.state === "booting") {
				await sleep(ENGINE_POLL_MS);
				continue;
			}
			return status;
		} catch (err) {
			// The engine handler may not be registered yet (background still
			// creating the offscreen document) — keep polling until the bound.
			lastError = err instanceof Error ? err.message : String(err);
			await sleep(ENGINE_POLL_MS);
		}
	}
	return { state: "failed", backend: null, error: lastError };
}

// Every queued request shares one readiness wait instead of polling per item.
// A failed readiness clears itself so the next request re-polls — the engine
// may still come up later.
let readiness: Promise<EngineStatus> | null = null;

export function waitForEngine(timeoutMs = ENGINE_WAIT_MS): Promise<EngineStatus> {
	readiness ??= pollEngine(timeoutMs).then((status) => {
		if (status.state === "failed") readiness = null;
		return status;
	});
	return readiness;
}

/** Reset the shared readiness wait; exported for tests and future reuse. */
export function resetEngineWait(): void {
	readiness = null;
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
	return Promise.race([
		promise,
		sleep(timeoutMs).then(() => {
			throw new Error(`timed out after ${timeoutMs}ms`);
		}),
	]);
}

/**
 * Classify an image URL. `frameUrl` carries a JPEG data URL of the pixels when
 * the URL itself is not fetchable by the engine (`data:`/`blob:`, or a video
 * element snapshot). Waits for the engine to boot, bounded at `timeoutMs`.
 */
export async function classifyImage(
	input: { url: string; width: number; height: number; frameUrl?: string },
	timeoutMs = ENGINE_WAIT_MS,
): Promise<Verdict> {
	const status = await waitForEngine(timeoutMs);
	if (status.state === "failed") {
		throw new EngineUnavailableError(status);
	}
	return withTimeout(sendMessage("classifyImage", input), timeoutMs);
}
