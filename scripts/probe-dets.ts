/**
 * One-off probe: classify specific fixtures through the built extension and
 * print FULL detection reasons (label, score, box) — eval-compare strips boxes.
 * Usage: pnpm exec tsx scripts/probe-dets.ts <file...>
 */
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { extname, join } from "node:path";
import { chromium } from "@playwright/test";

const EXTENSION_DIR = ".output/chrome-mv3";
const FIXTURES_DIR = "tests/fixtures";
const MIME: Record<string, string> = {
	".jpg": "image/jpeg",
	".jpeg": "image/jpeg",
	".png": "image/png",
	".webp": "image/webp",
};

declare const chrome: {
	runtime: {
		sendMessage(message: unknown, callback: (response: unknown) => void): void;
		lastError?: { message?: string };
	};
};

function serveFixtures() {
	const server = createServer((req, res) => {
		const name = decodeURIComponent((req.url ?? "/").split("?")[0]?.replace(/^\/+/, "") ?? "/");
		const file = join(FIXTURES_DIR, name);
		if (!file.startsWith(FIXTURES_DIR) || !existsSync(file) || name.includes("..")) {
			res.writeHead(404);
			res.end();
			return;
		}
		res.writeHead(200, { "content-type": MIME[extname(name)] ?? "application/octet-stream" });
		readFile(file).then((buf) => res.end(buf));
	});
	return new Promise<{ url: string; close: () => Promise<void> }>((resolve) => {
		server.listen(0, "127.0.0.1", () => {
			const addr = server.address();
			const port = typeof addr === "object" && addr != null ? addr.port : 0;
			resolve({
				url: `http://127.0.0.1:${port}`,
				close: () => new Promise((d) => server.close(() => d())),
			});
		});
	});
}

async function main() {
	const files = process.argv.slice(2);
	if (files.length === 0) throw new Error("usage: probe-dets.ts <file...>");
	const { url: base, close } = await serveFixtures();
	const userDataDir = await mkdtemp(join(tmpdir(), "hb-probe-"));
	const context = await chromium.launchPersistentContext(userDataDir, {
		headless: false,
		args: [`--disable-extensions-except=${EXTENSION_DIR}`, `--load-extension=${EXTENSION_DIR}`],
	});
	try {
		let worker = context.serviceWorkers().find((sw) => sw.url().startsWith("chrome-extension://"));
		worker ??= await context.waitForEvent("serviceworker", { timeout: 15_000 });
		const extensionId = new URL(worker.url()).host;
		const page = await context.newPage();
		await page.goto(`chrome-extension://${extensionId}/popup.html`);
		for (const file of files) {
			const res = await page.evaluate(
				(url) =>
					new Promise((resolve, reject) =>
						chrome.runtime.sendMessage(
							{ id: 1, type: "classifyImage", data: { url }, timestamp: Date.now() },
							(resp: unknown) => {
								const lastError = chrome.runtime.lastError;
								if (lastError) reject(new Error(lastError.message));
								else resolve(resp);
							},
						),
					),
				`${base}/${file}`,
			);
			const v = (res as { res?: { decision?: string; reasons?: unknown[]; ms?: number } }).res;
			console.log(`\n=== ${file} → ${v?.decision} (${v?.ms}ms)`);
			for (const d of v?.reasons ?? []) console.log("  ", JSON.stringify(d));
		}
	} finally {
		await context.close();
		await close();
		await rm(userDataDir, { recursive: true, force: true });
	}
}

main().catch((e) => {
	console.error(e);
	process.exit(1);
});
