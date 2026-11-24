/**
 * Chrome engine host: an offscreen document that owns the inference Worker
 * (MV3 service workers cannot spawn one). Firefox has no offscreen API — the
 * same `createEngineHost` code runs inside the background page there.
 */
import { browser } from "wxt/browser";
import { createEngineHost } from "../../engine/host";
import { onMessage } from "../../shared/protocol";

// Host construction must never take the message listeners down with it —
// a boot failure is reported through engineHostStatus instead.
let host: ReturnType<typeof createEngineHost> | null = null;
let bootError: string | null = null;
try {
	host = createEngineHost({
		modelUrl: browser.runtime.getURL("/models/320n.onnx"),
		faceModelUrl: browser.runtime.getURL("/models/fastface-small-112.onnx"),
		wasmBase: `${browser.runtime.getURL("/")}ort/`,
	});
} catch (e) {
	bootError = String(e);
}

// Internal protocol — reached only via the background's forwarding.
onMessage("engineRun", ({ data }) =>
	host == null ? { error: `host boot failed: ${bootError}` } : host.run(data),
);
onMessage("engineHostStatus", () =>
	host == null
		? { state: "failed", backend: null, error: `host boot failed: ${bootError}` }
		: host.status(),
);
