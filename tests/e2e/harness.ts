import { existsSync, readFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { type BrowserContext, chromium, type Page } from "@playwright/test";

/**
 * Shared E2E plumbing (WP5): a persistent Chromium context with the unpacked
 * extension, a localhost fixture server, and helpers to talk the
 * `@webext-core/messaging` wire protocol and `chrome.storage.sync` from an
 * extension page.
 */

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
export const EXTENSION_DIR = join(ROOT, ".output", "chrome-mv3");
export const FIXTURES_DIR = join(ROOT, "tests", "fixtures");

const MIME: Record<string, string> = {
	".html": "text/html; charset=utf-8",
	".png": "image/png",
	".jpg": "image/jpeg",
	".jpeg": "image/jpeg",
	".webp": "image/webp",
};

/** Minimal chrome surface used inside extension pages during E2E. */
export declare const chrome: {
	runtime: {
		sendMessage(message: unknown, callback: (response: WireResponse) => void): void;
		lastError?: { message?: string };
	};
	storage: {
		sync: {
			set(items: Record<string, unknown>, callback?: () => void): void;
		};
	};
	tabs: {
		query(
			info: Record<string, unknown>,
			callback: (tabs: { id?: number; url?: string }[]) => void,
		): void;
	};
};

export interface WireResponse {
	res?: unknown;
	err?: unknown;
}

/** Default settings — mirrors DEFAULT_SETTINGS in src/shared/settings.ts (v4). */
export const BASE_SETTINGS = {
	status: true,
	blurryStartMode: true,
	blurAmount: 20,
	gray: true,
	strictness: 0.5,
	blurImages: true,
	blurVideos: true,
	blurMale: false,
	blurFemale: true,
	strictModesty: false,
	skipChildren: true,
	unblurImages: false,
	unblurVideos: false,
	allowlist: [] as string[],
	failClosed: true,
	regionBlur: true,
};

/** Launch Chromium with the unpacked extension loaded. */
export async function launchExtension(lang = "en"): Promise<{
	context: BrowserContext;
	extensionId: string;
	userDataDir: string;
}> {
	const userDataDir = await mkdtemp(join(tmpdir(), "hb-e2e-"));
	const context = await chromium.launchPersistentContext(userDataDir, {
		headless: false,
		args: [
			`--lang=${lang}`,
			`--disable-extensions-except=${EXTENSION_DIR}`,
			`--load-extension=${EXTENSION_DIR}`,
		],
	});
	let worker = context.serviceWorkers().find((sw) => sw.url().startsWith("chrome-extension://"));
	worker ??= await context.waitForEvent("serviceworker", { timeout: 15_000 });
	return { context, extensionId: new URL(worker.url()).host, userDataDir };
}

export async function closeExtension(context: BrowserContext, userDataDir: string): Promise<void> {
	await context.close();
	await rm(userDataDir, { recursive: true, force: true });
}

/**
 * Serve `tests/fixtures/` plus the E2E pages below on 127.0.0.1.
 * `/` and `/page.html` render the main fixture page; `/frame.html` is the
 * iframe's document.
 */
export function serveFixtures(pages: Record<string, string>): Promise<{
	url: string;
	close: () => Promise<void>;
}> {
	const server: Server = createServer((req, res) => {
		const name = decodeURIComponent((req.url ?? "/").split("?")[0] ?? "/");
		const page = pages[name];
		if (page != null) {
			res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
			res.end(page);
			return;
		}
		const file = join(FIXTURES_DIR, name.replace(/^\/+/, ""));
		if (!file.startsWith(FIXTURES_DIR) || !existsSync(file) || name.includes("..")) {
			res.writeHead(404);
			res.end();
			return;
		}
		res.writeHead(200, { "content-type": MIME[extname(name)] ?? "application/octet-stream" });
		res.end(readFileSync(file));
	});
	return new Promise((resolve) => {
		server.listen(0, "127.0.0.1", () => {
			const addr = server.address();
			const port = typeof addr === "object" && addr != null ? addr.port : 0;
			resolve({
				url: `http://127.0.0.1:${port}`,
				close: () => new Promise((done) => server.close(() => done())),
			});
		});
	});
}

/**
 * `sendMessage(name, data)` over the messaging wire format, evaluated inside
 * an extension page (`chrome.runtime.sendMessage`).
 */
export function sendProtoMessage<T>(page: Page, type: string, data?: unknown): Promise<T> {
	return page.evaluate(
		([t, d]) =>
			new Promise<WireResponse>((resolve, reject) =>
				chrome.runtime.sendMessage(
					{ id: 1, type: t, data: d, timestamp: Date.now() },
					(resp: WireResponse) => {
						const lastError = chrome.runtime.lastError;
						if (lastError) reject(new Error(lastError.message));
						else resolve(resp);
					},
				),
			).then((env) => {
				if (env.err != null) throw new Error(`protocol error: ${JSON.stringify(env.err)}`);
				return env.res as T;
			}),
		[type, data],
	) as Promise<T>;
}

/** Merge a patch over the default settings into chrome.storage.sync (v2). */
export function writeSettings(page: Page, patch: Record<string, unknown>): Promise<void> {
	return page.evaluate(
		([base, p]) =>
			new Promise<void>((resolve) => {
				chrome.storage.sync.set(
					{ "hb-settings": { ...base, ...p }, "hb-settings$": { v: 2 } },
					() => resolve(),
				);
			}),
		[BASE_SETTINGS, patch],
	);
}

/** The id of the tab currently showing `urlPrefix`, seen from an extension page. */
export function findTabId(page: Page, urlPrefix: string): Promise<number | null> {
	return page.evaluate(
		(prefix) =>
			new Promise<number | null>((resolve) => {
				chrome.tabs.query({}, (tabs) => {
					const tab = tabs.find((t) => typeof t.url === "string" && t.url.startsWith(prefix));
					resolve(tab?.id ?? null);
				});
			}),
		urlPrefix,
	);
}
