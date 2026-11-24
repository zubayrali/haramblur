import { type Browser, browser } from "wxt/browser";
import { defineBackground } from "wxt/utils/define-background";
import { t } from "../shared/i18n";
import { type ContextMediaKind, onMessage, sendMessage, type TabStats } from "../shared/protocol";

const OFFSCREEN_DOCUMENT_URL = "/offscreen.html";
const REASON_WORKERS = "WORKERS" as Browser.offscreen.Reason;

const MENU_REVEAL = "hb-reveal-media";
const MENU_RECHECK = "hb-recheck-media";

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

const emptyStats = (): TabStats => ({ total: 0, images: 0, videos: 0, byState: {} });

/** Runs inside each page frame; must stay free of outer-scope references. */
function countFrameMedia(): TabStats {
	const stats: TabStats = { total: 0, images: 0, videos: 0, byState: {} };
	for (const el of document.querySelectorAll("[data-hb-state]")) {
		const state = el.getAttribute("data-hb-state") ?? "unknown";
		stats.byState[state] = (stats.byState[state] ?? 0) + 1;
		stats.total += 1;
		if (el.tagName === "IMG") stats.images += 1;
		else if (el.tagName === "VIDEO") stats.videos += 1;
	}
	return stats;
}

/** Count elements carrying `data-hb-state` across all frames of a tab. */
async function getTabStats(tabId: number): Promise<TabStats> {
	const scripting = browser.scripting;
	if (scripting == null) return emptyStats();
	try {
		const results = await scripting.executeScript({
			target: { tabId, allFrames: true },
			func: countFrameMedia,
		});
		const merged = emptyStats();
		for (const frame of results) {
			const s = frame.result;
			if (s == null) continue;
			merged.total += s.total;
			merged.images += s.images;
			merged.videos += s.videos;
			for (const [state, count] of Object.entries(s.byState)) {
				merged.byState[state] = (merged.byState[state] ?? 0) + count;
			}
		}
		return merged;
	} catch {
		// Tab not scriptable (chrome://, store pages, extension pages…).
		return emptyStats();
	}
}

function createContextMenus(): void {
	browser.contextMenus
		.removeAll()
		.then(() => {
			browser.contextMenus.create({
				id: MENU_REVEAL,
				title: t("revealThisMedia"),
				contexts: ["image", "video"],
			});
			browser.contextMenus.create({
				id: MENU_RECHECK,
				title: t("recheckThisMedia"),
				contexts: ["image", "video"],
			});
		})
		.catch((err) => console.error("[haramblur] context menu setup failed", err));
}

async function forwardMediaAction(
	action: "revealMedia" | "recheckMedia",
	info: Browser.contextMenus.OnClickData,
	tab: Browser.tabs.Tab | undefined,
): Promise<void> {
	const srcUrl = info.srcUrl;
	const tabId = tab?.id;
	if (srcUrl == null || tabId == null) return;
	const kind: ContextMediaKind = info.mediaType === "video" ? "video" : "image";
	try {
		await sendMessage(action, { srcUrl, kind }, { tabId });
	} catch {
		// WP2's content script is not installed on this tab (or not landed yet).
		console.debug(`[haramblur] ${action}: no listener in tab ${tabId}`);
	}
}

export default defineBackground({
	persistent: false,
	main() {
		onMessage("ping", () => ({ pong: true, ts: Date.now() }));
		onMessage("getTabStats", ({ data }) => getTabStats(data.tabId));
		createContextMenus();
		browser.contextMenus.onClicked.addListener((info, tab) => {
			const action =
				info.menuItemId === MENU_REVEAL
					? "revealMedia"
					: info.menuItemId === MENU_RECHECK
						? "recheckMedia"
						: null;
			if (action != null) void forwardMediaAction(action, info, tab);
		});
		ensureOffscreenDocument().catch((err) =>
			console.error("[haramblur] offscreen creation failed", err),
		);
	},
});
