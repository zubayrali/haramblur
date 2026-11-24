/**
 * Engine worker stub. WP1 replaces this with the XState lifecycle +
 * onnxruntime-web NudeNet runner.
 */
interface WorkerScope {
	onmessage: ((event: MessageEvent<{ type?: string }>) => void) | null;
	postMessage(data: unknown): void;
}

const scope = self as unknown as WorkerScope;

scope.onmessage = (event) => {
	if (event.data?.type === "ping") {
		scope.postMessage({ pong: true, ts: Date.now() });
	}
};

export {};
