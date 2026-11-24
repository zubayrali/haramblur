import { defineContentScript } from "wxt/utils/define-content-script";
import { startContentPipeline } from "../../content/index";

/**
 * WP2 content pipeline entrypoint. Runs in every frame on http(s) pages at
 * document_start so iframes are covered for free and media is blurred before
 * it renders (fail closed).
 */
export default defineContentScript({
	matches: ["http://*/*", "https://*/*"],
	runAt: "document_start",
	allFrames: true,
	async main() {
		try {
			await startContentPipeline();
		} catch (err) {
			console.error("[haramblur] content pipeline failed to start", err);
		}
	},
});
