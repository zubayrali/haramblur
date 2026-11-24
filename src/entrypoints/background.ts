import { type Browser, browser } from "wxt/browser";
import { defineBackground } from "wxt/utils/define-background";
import { onMessage } from "../shared/protocol";

const OFFSCREEN_DOCUMENT_URL = "/offscreen.html";
const REASON_WORKERS = "WORKERS" as Browser.offscreen.Reason;

/**
 * Chrome hosts the ML engine in an offscreen document so it can spawn a
 * dedicated Worker (MV3 service workers can't). Firefox builds have no
 * offscreen API — WP1 hosts the engine in the background page there instead
 * (docs/PLAN.md "Engine host").
 */
async function ensureOffscreenDocument(): Promise<void> {
	const offscreen = browser.offscreen;
	if (offscreen == null) return;
	if (await offscreen.hasDocument()) return;
	await offscreen.createDocument({
		url: browser.runtime.getURL(OFFSCREEN_DOCUMENT_URL),
		reasons: [REASON_WORKERS],
		justification: "Run on-device ML inference (onnxruntime-web) in a dedicated worker.",
	});
}

export default defineBackground({
	persistent: false,
	main() {
		onMessage("ping", () => ({ pong: true, ts: Date.now() }));
		ensureOffscreenDocument().catch((err) =>
			console.error("[haramblur] offscreen creation failed", err),
		);
	},
});
