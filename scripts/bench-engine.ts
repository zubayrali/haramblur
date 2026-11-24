/**
 * `pnpm bench` — end-to-end engine benchmark in Chromium with the built
 * extension loaded (docs/PLAN.md acceptance: load < 2 s, < 150 ms/image).
 *
 * Serves `tests/fixtures/` over localhost, drives the engine through the real
 * `classifyImage`/`engineStatus` protocol messages, and prints model load ms
 * plus per-image ms. Requires `pnpm build` first; run under `xvfb-run` without
 * a display (extensions need a headed browser).
 */

import { existsSync } from "node:fs";
import { mkdtemp, readdir, readFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const EXTENSION_DIR = join(ROOT, ".output", "chrome-mv3");
const FIXTURES_DIR = join(ROOT, "tests", "fixtures");

const LOAD_BUDGET_MS = 2_000;
const PER_IMAGE_BUDGET_MS = 150;
const BENCH_PASSES = 3;

const MIME: Record<string, string> = {
	".png": "image/png",
	".jpg": "image/jpeg",
	".jpeg": "image/jpeg",
	".webp": "image/webp",
};

interface MessageResponse {
	res?: unknown;
	err?: unknown;
}

/** Minimal chrome runtime shape used inside `page.evaluate` (extension page). */
declare const chrome: {
	runtime: {
		sendMessage(message: unknown, callback: (response: MessageResponse) => void): void;
		lastError?: { message?: string };
	};
};

function serveFixtures(): Promise<{ url: string; close: () => Promise<void> }> {
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

async function main(): Promise<void> {
	if (!existsSync(EXTENSION_DIR)) {
		throw new Error(`${EXTENSION_DIR} missing — run \`pnpm build\` first`);
	}
	const fixtures = (await readdir(FIXTURES_DIR)).filter((f) => extname(f) in MIME);
	if (fixtures.length === 0) throw new Error("no fixture images under tests/fixtures/");

	const { url: base, close } = await serveFixtures();
	const userDataDir = await mkdtemp(join(tmpdir(), "hb-bench-"));
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

		const send = (type: string, data?: unknown) =>
			page.evaluate(
				([t, d]) =>
					new Promise<MessageResponse>((resolve, reject) =>
						chrome.runtime.sendMessage(
							{ id: 1, type: t, data: d, timestamp: Date.now() },
							(resp: MessageResponse) => {
								const lastError = chrome.runtime.lastError;
								if (lastError) reject(new Error(lastError.message));
								else resolve(resp);
							},
						),
					),
				[type, data],
			) as Promise<MessageResponse>;

		const call = async <T>(type: string, data?: unknown): Promise<T> => {
			const env = await send(type, data);
			if (env.err != null) throw new Error(`protocol error: ${JSON.stringify(env.err)}`);
			return env.res as T;
		};

		// Model load: from the first engineStatus probe until the machine leaves
		// booting. The background creates the host eagerly at startup, so the
		// load overlaps SW wake — measure wall time here.
		const t0 = Date.now();
		let status: { state: string; backend: string | null; error?: string };
		do {
			status = await call("engineStatus");
			await new Promise((r) => setTimeout(r, 50));
		} while (status.state === "booting" && Date.now() - t0 < 60_000);
		const loadMs = Date.now() - t0;
		console.log(`engine: state=${status.state} backend=${status.backend} load=${loadMs}ms`);
		if (status.state !== "ready" && status.state !== "degraded") {
			throw new Error(`engine never became ready: ${status.error ?? status.state}`);
		}

		// One warmup pass: the first inference compiles WASM kernels and is not
		// representative of steady-state per-image cost.
		const warmUrl = `${base}/${fixtures[0]}?warm=1`;
		const warmDims = await page.evaluate(async (u) => {
			const bmp = await createImageBitmap(await (await fetch(u)).blob());
			return { width: bmp.width, height: bmp.height };
		}, warmUrl);
		await call("classifyImage", { url: warmUrl, width: warmDims.width, height: warmDims.height });

		const perImage: number[] = [];
		const perImageEngine: number[] = [];
		for (let pass = 0; pass < BENCH_PASSES; pass++) {
			for (const name of fixtures) {
				const url = `${base}/${name}?pass=${pass}`;
				const dims = await page.evaluate(async (u) => {
					const bmp = await createImageBitmap(await (await fetch(u)).blob());
					return { width: bmp.width, height: bmp.height };
				}, url);
				const started = Date.now();
				const verdict = await call<{
					decision?: string;
					reasons?: { label: string; score: number; box: number[] }[];
					ms?: number;
					cached?: boolean;
					error?: string;
				}>("classifyImage", { url, width: dims.width, height: dims.height });
				const wallMs = Date.now() - started;
				if (verdict.error != null) {
					console.log(`${name}: ERROR ${verdict.error} (${wallMs}ms)`);
					continue;
				}
				perImage.push(wallMs);
				if (verdict.ms != null) perImageEngine.push(verdict.ms);
				const top = (verdict.reasons ?? [])
					.slice(0, 3)
					.map((d) => `${d.label}@${d.score.toFixed(2)}`)
					.join(", ");
				console.log(
					`pass${pass} ${name}: ${verdict.decision} wall=${wallMs}ms engine=${verdict.ms}ms [${top}]`,
				);
			}
		}

		// Cached replay: second call must hit the LRU cache.
		const repeat = await call<{ cached?: boolean; ms?: number; error?: string }>("classifyImage", {
			url: `${base}/${fixtures[0]}?pass=0`,
			width: 1,
			height: 1,
		});
		console.log(`cached replay: cached=${repeat.cached} ms=${repeat.ms}`);

		const stats = (xs: number[]) => ({
			mean: xs.reduce((a, b) => a + b, 0) / Math.max(xs.length, 1),
			max: Math.max(...xs),
			n: xs.length,
		});
		const wall = stats(perImage);
		const engine = stats(perImageEngine);
		// Budgets apply to the engine's reported inference+policy ms (the
		// "ms/image" in ENGINE_NOTES.md); wall adds the message round-trip.
		console.log(
			`summary: load=${loadMs}ms engine.mean=${engine.mean.toFixed(0)}ms engine.max=${engine.max}ms wall.mean=${wall.mean.toFixed(0)}ms wall.max=${wall.max}ms n=${engine.n}`,
		);
		let failed = false;
		if (loadMs >= LOAD_BUDGET_MS) {
			console.error(`FAIL: load ${loadMs}ms >= ${LOAD_BUDGET_MS}ms`);
			failed = true;
		}
		if (engine.mean >= PER_IMAGE_BUDGET_MS) {
			console.error(`FAIL: engine mean ${engine.mean.toFixed(0)}ms >= ${PER_IMAGE_BUDGET_MS}ms`);
			failed = true;
		}
		if (engine.max >= PER_IMAGE_BUDGET_MS * 2) {
			console.error(`FAIL: engine max ${engine.max}ms >= ${PER_IMAGE_BUDGET_MS * 2}ms`);
			failed = true;
		}
		if (repeat.cached !== true) {
			console.error("FAIL: second classifyImage did not hit the verdict cache");
			failed = true;
		}
		if (failed) process.exit(1);
	} finally {
		await context.close();
		await rm(userDataDir, { recursive: true, force: true });
		await close();
	}
}

main().catch((err) => {
	console.error(err instanceof Error ? err.message : err);
	process.exit(1);
});
