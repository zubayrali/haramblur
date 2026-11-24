/**
 * Chrome engine host (WP1 owns the real engine). The stub verifies the Worker
 * wiring: it spawns a module worker and round-trips a ping.
 */
declare const self: Window;

const worker = new Worker(new URL("./worker.ts", import.meta.url), {
	type: "module",
});
worker.onmessage = (event: MessageEvent) => {
	console.debug("[haramblur] engine worker replied:", event.data);
};
worker.onerror = (event) => {
	console.error("[haramblur] engine worker failed:", event);
};
worker.postMessage({ type: "ping" });
