/**
 * `pnpm eval` — accuracy eval harness (docs/PLAN.md WP5, "eval script that
 * scores the engine on a labelled fixture set").
 *
 * Serves `tests/fixtures/` over localhost, drives the real engine through the
 * `classifyImage` protocol inside Chromium with the unpacked extension loaded,
 * and compares every fixture against `tests/fixtures/manifest.json`:
 *
 *   - per-image verdict (expected vs actual decision, with detections)
 *   - per-class precision/recall (a class counts predicted when detected at
 *     or above --threshold, default 0.3)
 *
 * Requires `pnpm build` first; run under `xvfb-run` without a display.
 */

import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const EXTENSION_DIR = join(ROOT, ".output", "chrome-mv3");
const FIXTURES_DIR = join(ROOT, "tests", "fixtures");
const MANIFEST_PATH = join(FIXTURES_DIR, "manifest.json");

const MIME: Record<string, string> = {
	".png": "image/png",
	".jpg": "image/jpeg",
	".jpeg": "image/jpeg",
	".webp": "image/webp",
};

interface ManifestEntry {
	expect: string[];
	decision: "blur" | "clear";
}

interface Manifest {
	files: Record<string, ManifestEntry>;
}

interface WireResponse {
	res?: unknown;
	err?: unknown;
}

interface Detection {
	label: string;
	score: number;
	box: [number, number, number, number];
}

interface Verdict {
	decision?: string;
	reasons?: Detection[];
	ms?: number;
	error?: string;
}

declare const chrome: {
	runtime: {
		sendMessage(message: unknown, callback: (response: WireResponse) => void): void;
		lastError?: { message?: string };
	};
};

function thresholdArg(): number {
	const i = process.argv.indexOf("--threshold");
	if (i < 0) return 0.3;
	const v = Number(process.argv[i + 1]);
	return Number.isFinite(v) ? v : 0.3;
}

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
	const manifest = JSON.parse(await readFile(MANIFEST_PATH, "utf8")) as Manifest;
	const threshold = thresholdArg();

	const { url: base, close } = await serveFixtures();
	const userDataDir = await mkdtemp(join(tmpdir(), "hb-eval-"));
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

		const call = async <T>(type: string, data?: unknown): Promise<T> => {
			const env = await page.evaluate(
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
					),
				[type, data],
			);
			if (env.err != null) throw new Error(`protocol error: ${JSON.stringify(env.err)}`);
			return env.res as T;
		};

		// Wait out the boot once; per-image calls then measure steady state.
		for (let i = 0; i < 240; i++) {
			const s = await call<{ state: string }>("engineStatus");
			if (s.state !== "booting") break;
			await new Promise((r) => setTimeout(r, 250));
		}

		console.log(`eval: ${Object.keys(manifest.files).length} fixtures, threshold=${threshold}\n`);

		// Per-class TP/FP/FN tallies and decision accuracy.
		const classes = new Set<string>();
		for (const entry of Object.values(manifest.files)) {
			for (const label of entry.expect) classes.add(label);
		}
		const tally = new Map<string, { tp: number; fp: number; fn: number }>();
		for (const c of classes) tally.set(c, { tp: 0, fp: 0, fn: 0 });
		let correctDecisions = 0;

		for (const [name, entry] of Object.entries(manifest.files)) {
			const url = `${base}/${name}`;
			const dims = await page.evaluate(async (u) => {
				const bmp = await createImageBitmap(await (await fetch(u)).blob());
				return { width: bmp.width, height: bmp.height };
			}, url);
			const verdict = await call<Verdict>("classifyImage", {
				url,
				width: dims.width,
				height: dims.height,
			});

			if (verdict.error != null) {
				console.log(`${name}: ERROR ${verdict.error}`);
				for (const label of entry.expect) tally.get(label)!.fn += 1;
				continue;
			}

			const predicted = new Set(
				(verdict.reasons ?? []).filter((d) => d.score >= threshold).map((d) => d.label),
			);
			const decisionOk = verdict.decision === entry.decision;
			if (decisionOk) correctDecisions += 1;

			for (const label of entry.expect) {
				const t = tally.get(label)!;
				if (predicted.has(label)) t.tp += 1;
				else t.fn += 1;
			}
			for (const label of predicted) {
				if (!entry.expect.includes(label)) {
					const t = tally.get(label) ?? { tp: 0, fp: 0, fn: 0 };
					t.fp += 1;
					tally.set(label, t);
					classes.add(label);
				}
			}

			const top = (verdict.reasons ?? [])
				.slice(0, 4)
				.map((d) => `${d.label}@${d.score.toFixed(2)}`)
				.join(", ");
			console.log(
				`${name}: ${verdict.decision} (expected ${entry.decision}${decisionOk ? "" : " ✗"}) [${top}]`,
			);
		}

		console.log(`\ndecision accuracy: ${correctDecisions}/${Object.keys(manifest.files).length}`);
		console.log(`\nper-class precision/recall (detections >= ${threshold}):`);
		for (const cls of [...classes].sort()) {
			const t = tally.get(cls)!;
			const precision = t.tp + t.fp > 0 ? t.tp / (t.tp + t.fp) : null;
			const recall = t.tp + t.fn > 0 ? t.tp / (t.tp + t.fn) : null;
			const fmt = (v: number | null) =>
				v == null ? "  —  " : `${(v * 100).toFixed(0).padStart(4)}%`;
			console.log(
				`  ${cls}: precision=${fmt(precision)} recall=${fmt(recall)} (tp=${t.tp} fp=${t.fp} fn=${t.fn})`,
			);
		}
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
