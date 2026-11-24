import { type Browser, browser } from "wxt/browser";
import { defineBackground } from "wxt/utils/define-background";
import { createEngineHost, type EngineHost } from "../engine/host";
import { t } from "../shared/i18n";
import {
	type ClassifyResult,
	type ContextMediaKind,
	type EngineRequest,
	type EngineStatus,
	isEngineError,
	onMessage,
	sendMessage,
	type TabStats,
	type Verdict,
} from "../shared/protocol";
import { getSettings } from "../shared/settings-store";

const OFFSCREEN_DOCUMENT_URL = "/offscreen.html";
const REASON_WORKERS = "WORKERS" as Browser.offscreen.Reason;
const HOST_PROBE_ATTEMPTS = 30;
const HOST_PROBE_DELAY_MS = 150;

const MENU_REVEAL = "hb-reveal-media";
const MENU_RECHECK = "hb-recheck-media";

/**
 * How the background reaches the engine host (docs/PLAN.md "Engine host"):
 * Chrome runs it in an offscreen document; Firefox runs the same host code
 * inside the background page (`browser.offscreen` does not exist there).
 */
interface EngineClient {
	run(request: EngineRequest): Promise<ClassifyResult>;
	status(): Promise<EngineStatus>;
}

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

/** Wait until the offscreen host's message listener answers a status probe. */
async function probeHost(): Promise<void> {
	for (let i = 0; i < HOST_PROBE_ATTEMPTS; i++) {
		try {
			const status = await sendMessage("engineHostStatus");
			if (typeof status?.state === "string") return;
		} catch {
			// Host page is still loading its listener.
		}
		await new Promise((resolve) => setTimeout(resolve, HOST_PROBE_DELAY_MS));
	}
	throw new Error("engine host did not come up");
}

function chromeClient(): EngineClient {
	let ready: Promise<void> | null = null;
	const ensure = async (): Promise<void> => {
		if (ready != null) return ready;
		ready = ensureOffscreenDocument().then(probeHost);
		try {
			await ready;
		} catch (err) {
			ready = null;
			throw err;
		}
	};
	return {
		async run(request) {
			try {
				await ensure();
			} catch (err) {
				return { error: `engine host unreachable: ${errMsg(err)}` };
			}
			try {
				return await sendMessage("engineRun", request);
			} catch (err) {
				ready = null;
				return { error: `engine request failed: ${errMsg(err)}` };
			}
		},
		async status() {
			try {
				await ensure();
				return await sendMessage("engineHostStatus");
			} catch (err) {
				return {
					state: "failed",
					backend: null,
					error: `engine host unreachable: ${errMsg(err)}`,
				};
			}
		},
	};
}

function firefoxClient(): EngineClient {
	let host: EngineHost | null = null;
	const get = (): EngineHost =>
		(host ??= createEngineHost({
			modelUrl: browser.runtime.getURL("/models/320n.onnx"),
			faceModelUrl: browser.runtime.getURL("/models/fastface-small-112.onnx"),
			faceDetModelUrl: browser.runtime.getURL("/models/blaze-face.onnx"),
			wasmBase: `${browser.runtime.getURL("/")}ort/`,
		}));
	return {
		run: (request) => get().run(request),
		status: async () => get().status(),
	};
}

const errMsg = (err: unknown) => (err instanceof Error ? err.message : String(err));

/**
 * Public classify calls answer `Verdict`; on engine failure the background
 * throws, which `sendMessage` surfaces as a rejection — the caller (WP2's
 * content client) applies `settings.failClosed` to it.
 */
async function classify(engine: EngineClient, request: EngineRequest): Promise<Verdict> {
	const result = await engine.run(request);
	if (isEngineError(result)) throw new Error(result.error);
	return result;
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
	// Firefox runs the engine host inside the background page itself; keeping
	// the page persistent keeps the loaded model resident (an event page would
	// unload it between messages, forcing a cold boot per request burst).
	// MV3 ignores the flag — service workers are never persistent.
	persistent: { firefox: true },
	main() {
		const engine = browser.offscreen == null ? firefoxClient() : chromeClient();

		onMessage("ping", () => ({ pong: true, ts: Date.now() }));
		// Offscreen documents cannot touch chrome.storage, so the background
		// reads settings itself and sends them along with each request.
		onMessage("classifyImage", async ({ data }) =>
			classify(engine, {
				kind: "image",
				url: data.url,
				width: data.width,
				height: data.height,
				frameUrl: data.frameUrl,
				settings: await getSettings(),
			}),
		);
		onMessage("classifyFrame", async ({ data }) =>
			classify(engine, {
				kind: "frame",
				dataUrl: data.dataUrl,
				videoId: data.videoId,
				ts: data.ts,
				settings: await getSettings(),
			}),
		);
		onMessage("engineStatus", () => engine.status());
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

		if (browser.offscreen == null) {
			// Firefox: create the in-process host eagerly so the model warms up now.
			engine.status().catch(() => undefined);
		} else {
			ensureOffscreenDocument().catch((err) =>
				console.error("[haramblur] offscreen creation failed", err),
			);
		}
	},
});
