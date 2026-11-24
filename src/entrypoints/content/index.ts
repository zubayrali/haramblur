import { defineContentScript } from "wxt/utils/define-content-script";

/**
 * WP2 owns discovery/queue/blur. This stub only pins the contract: every frame
 * on http(s) pages, at document_start.
 */
export default defineContentScript({
	matches: ["http://*/*", "https://*/*"],
	runAt: "document_start",
	allFrames: true,
	main() {},
});
