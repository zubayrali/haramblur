/**
 * `pnpm exec tsx scripts/eval-compare.ts` — WP-B benchmark harness.
 *
 * Scores the built extension (`.output/chrome-mv3`) against
 * `tests/fixtures/manifest.v2.json` (dev/test splits + categories) and prints
 * accuracy / precision / recall / F1 / false-blur / miss rates + p50/p95
 * latency, overall and per category.
 *
 *   --split dev|test|all   which manifest split to score (default: test)
 *   --baseline <file.json> compare against a previous results file
 *   --label <name>         label for this run in the output header
 *   --out <file.json>      write raw per-image results JSON (baseline input)
 *
 * The original 0.2.x extension is MV2 and cannot load in Playwright Chromium.
 * To compare head-to-head, produce its results JSON with a Node port of its
 * detector (src/modules/detector.js @ 50d8943~1) writing the same schema,
 * then pass it via --baseline.
 *
 * Requires `pnpm build` first.
 */

import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const EXTENSION_DIR = join(ROOT, ".output", "chrome-mv3");
const FIXTURES_DIR = join(ROOT, "tests", "fixtures");
const MANIFEST_V2 = join(FIXTURES_DIR, "manifest.v2.json");

const MIME: Record<string, string> = {
	".png": "image/png",
	".jpg": "image/jpeg",
	".jpeg": "image/jpeg",
	".webp": "image/webp",
};

interface ManifestEntry {
	decision: "blur" | "clear";
	category: string;
	expect?: string[];
}
interface ManifestV2 {
	splits: Record<string, { files: Record<string, ManifestEntry> }>;
}
interface ImageResult {
	file: string;
	split: string;
	category: string;
	expected: "blur" | "clear";
	actual: "blur" | "clear" | "error";
	ms: number;
	detections: { label: string; score: number }[];
	expectLabels: string[];
}
interface ResultsFile {
	engine: string;
	generatedAt: string;
	results: ImageResult[];
}
interface WireResponse {
	res?: { decision?: string; reasons?: { label: string; score: number }[]; ms?: number };
	err?: unknown;
}

declare const chrome: {
	storage: {
		sync: {
			get(key: string, callback: (value: Record<string, unknown>) => void): void;
			set(value: Record<string, unknown>, callback?: () => void): void;
		};
	};
	runtime: {
		sendMessage(message: unknown, callback: (response: WireResponse) => void): void;
		lastError?: { message?: string };
	};
};

function arg(flag: string): string | undefined {
	const i = process.argv.indexOf(flag);
	return i >= 0 ? process.argv[i + 1] : undefined;
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

interface Metrics {
	n: number;
	acc: number;
	precision: number;
	recall: number;
	f1: number;
	falseBlur: number;
	miss: number;
	p50: number;
	p95: number;
}

function percentile(xs: number[], p: number): number {
	if (xs.length === 0) return 0;
	const s = [...xs].sort((a, b) => a - b);
	return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))]!;
}

function metrics(rs: ImageResult[]): Metrics {
	const n = rs.length;
	const tp = rs.filter((r) => r.expected === "blur" && r.actual === "blur").length;
	const fp = rs.filter((r) => r.expected === "clear" && r.actual === "blur").length;
	const fn = rs.filter((r) => r.expected === "blur" && r.actual !== "blur").length;
	const tn = rs.filter((r) => r.expected === "clear" && r.actual !== "blur").length;
	const pos = tp + fn;
	const neg = tn + fp;
	const precision = tp + fp === 0 ? 1 : tp / (tp + fp);
	const recall = pos === 0 ? 1 : tp / pos;
	const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
	const lat = rs.map((r) => r.ms).filter((m) => m > 0);
	return {
		n,
		acc: n === 0 ? 0 : (tp + tn) / n,
		precision,
		recall,
		f1,
		falseBlur: neg === 0 ? 0 : fp / neg,
		miss: pos === 0 ? 0 : fn / pos,
		p50: percentile(lat, 50),
		p95: percentile(lat, 95),
	};
}

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

interface ClassTally {
	tp: number;
	fp: number;
	fn: number;
}

/** Presence-level per-class metrics over the manifest's human labels (dev tuning only). */
function perClass(rows: ImageResult[], threshold = 0.3): Map<string, ClassTally> {
	const out = new Map<string, ClassTally>();
	const get = (label: string) => {
		const cur = out.get(label) ?? { tp: 0, fp: 0, fn: 0 };
		out.set(label, cur);
		return cur;
	};
	for (const row of rows) {
		const predicted = new Set(
			row.detections.filter((d) => d.score >= threshold).map((d) => d.label),
		);
		for (const label of row.expectLabels) {
			if (predicted.has(label)) get(label).tp++;
			else get(label).fn++;
		}
		for (const label of predicted) {
			if (!row.expectLabels.includes(label)) get(label).fp++;
		}
	}
	return out;
}

function printPerClass(rows: ImageResult[]): void {
	console.log("\n### per-class precision/recall (presence, detections >= 0.3)\n");
	console.log("| class | precision | recall | tp | fp | fn |");
	console.log("|---|---:|---:|---:|---:|---:|");
	for (const [label, t] of [...perClass(rows).entries()].sort()) {
		const precision = t.tp + t.fp === 0 ? null : t.tp / (t.tp + t.fp);
		const recall = t.tp + t.fn === 0 ? null : t.tp / (t.tp + t.fn);
		const fmt = (v: number | null) => (v == null ? "—" : pct(v));
		console.log(`| ${label} | ${fmt(precision)} | ${fmt(recall)} | ${t.tp} | ${t.fp} | ${t.fn} |`);
	}
}

function printTable(title: string, rows: [string, Metrics, Metrics?][]): void {
	console.log(`\n### ${title}\n`);
	console.log(
		"| set | n | acc | prec | rec | F1 | false-blur | miss | p50ms | p95ms |" +
			(rows.some((r) => r[2]) ? " vs-base acc/F1 |" : ""),
	);
	console.log(`|---|---|---|---|---|---|---|---|---|---|${rows.some((r) => r[2]) ? "---|" : ""}`);
	for (const [name, m, base] of rows) {
		const line = `| ${name} | ${m.n} | ${pct(m.acc)} | ${pct(m.precision)} | ${pct(m.recall)} | ${pct(m.f1)} | ${pct(m.falseBlur)} | ${pct(m.miss)} | ${m.p50} | ${m.p95} |`;
		console.log(base ? `${line} ${pct(base.acc)}/${pct(base.f1)} |` : line);
	}
}

/**
 * Seed an OPFS NudeNet override before the measured browser starts. The first
 * seeding browser is discarded; the measurement browser then boots with the
 * override already present. The file is fetched through the extension package
 * to avoid transferring 100 MB through Playwright's evaluate bridge.
 */
async function seedNudenetOverride(userDataDir: string, extensionFile: string): Promise<void> {
	const context = await chromium.launchPersistentContext(userDataDir, {
		headless: false,
		args: [`--disable-extensions-except=${EXTENSION_DIR}`, `--load-extension=${EXTENSION_DIR}`],
	});
	try {
		let worker = context.serviceWorkers().find((sw) => sw.url().startsWith("chrome-extension://"));
		worker ??= await context.waitForEvent("serviceworker", { timeout: 15_000 });
		const extensionId = new URL(worker.url()).host;
		const page = await context.newPage();
		await page.addInitScript("globalThis.__name = (fn) => fn");
		await page.goto(`chrome-extension://${extensionId}/popup.html`);
		await page.evaluate(async (file) => {
			const res = await fetch(
				(chrome.runtime as unknown as { getURL(path: string): string }).getURL(file),
			);
			if (!res.ok) throw new Error(`override fetch failed: HTTP ${res.status}`);
			const bytes = new Uint8Array(await res.arrayBuffer());
			const root = await (
				navigator.storage as unknown as {
					getDirectory(): Promise<{
						getDirectoryHandle(
							name: string,
							opts: { create: boolean },
						): Promise<{
							getFileHandle(
								name: string,
								opts: { create: boolean },
							): Promise<{
								createWritable(): Promise<{
									write(data: Uint8Array): Promise<void>;
									close(): Promise<void>;
								}>;
							}>;
						}>;
					}>;
				}
			).getDirectory();
			const dir = await root.getDirectoryHandle("hb-models", { create: true });
			const write = async (name: string, data: Uint8Array) => {
				const fh = await dir.getFileHandle(name, { create: true });
				const w = await fh.createWritable();
				await w.write(data);
				await w.close();
			};
			await write("nudenet.onnx", bytes);
			const meta = {
				sha256: "",
				size: bytes.length,
				installedAt: Date.now(),
				fileName: file.split("/").at(-1),
				inputSize: 640,
			};
			await write("nudenet.json", new TextEncoder().encode(JSON.stringify(meta)));
		}, extensionFile);
		await page.close();
	} finally {
		await context.close();
	}
}

async function main(): Promise<void> {
	if (!existsSync(EXTENSION_DIR)) throw new Error(`${EXTENSION_DIR} missing — run pnpm build`);
	const manifest = JSON.parse(await readFile(MANIFEST_V2, "utf8")) as ManifestV2;
	const splitArg = arg("--split") ?? "test";
	const coverageMode = arg("--coverage-mode");
	if (coverageMode != null && !["face", "head", "body"].includes(coverageMode)) {
		throw new Error(`invalid --coverage-mode "${coverageMode}"`);
	}
	const detectorInput =
		arg("--detector-input") == null ? undefined : Number(arg("--detector-input"));
	const nudenetOverride = arg("--nudenet-override");
	const strictness = arg("--strictness") == null ? undefined : Number(arg("--strictness"));
	if (detectorInput != null && (!Number.isInteger(detectorInput) || detectorInput % 32 !== 0)) {
		throw new Error(`invalid --detector-input "${detectorInput}"`);
	}
	const splits = splitArg === "all" ? Object.keys(manifest.splits) : [splitArg];
	const entries: { file: string; split: string; entry: ManifestEntry }[] = [];
	for (const s of splits) {
		const files = manifest.splits[s]?.files;
		if (files == null) throw new Error(`unknown split "${s}"`);
		for (const [file, entry] of Object.entries(files)) entries.push({ file, split: s, entry });
	}

	const { url: base, close } = await serveFixtures();
	const userDataDir = await mkdtemp(join(tmpdir(), "hb-cmp-"));
	if (nudenetOverride != null) await seedNudenetOverride(userDataDir, nudenetOverride);
	const context = await chromium.launchPersistentContext(userDataDir, {
		headless: false,
		args: [`--disable-extensions-except=${EXTENSION_DIR}`, `--load-extension=${EXTENSION_DIR}`],
	});
	const results: ImageResult[] = [];
	try {
		let worker = context.serviceWorkers().find((sw) => sw.url().startsWith("chrome-extension://"));
		worker ??= await context.waitForEvent("serviceworker", { timeout: 15_000 });
		const extensionId = new URL(worker.url()).host;
		const page = await context.newPage();
		await page.goto(`chrome-extension://${extensionId}/popup.html`);
		if (coverageMode != null || strictness != null) {
			await page.evaluate(
				([mode, strictnessOverride]) =>
					new Promise<void>((resolve) =>
						chrome.storage.sync.get("hb-settings", (all) => {
							const current = (all["hb-settings"] ?? {}) as Record<string, unknown>;
							chrome.storage.sync.set(
								{
									"hb-settings": {
										...current,
										...(mode == null ? {} : { coverageMode: mode }),
										...(strictnessOverride == null ? {} : { strictness: strictnessOverride }),
									},
								},
								() => resolve(),
							);
						}),
					),
				[coverageMode, strictness],
			);
		}

		for (const { file, split, entry } of entries) {
			const res = await page.evaluate(
				([url, inputSize]) =>
					new Promise<WireResponse>((resolve, reject) =>
						chrome.runtime.sendMessage(
							{
								id: 1,
								type: "classifyImage",
								data: { url, ...(inputSize == null ? {} : { detectorInputSize: inputSize }) },
								timestamp: Date.now(),
							},
							(resp: WireResponse) => {
								const lastError = chrome.runtime.lastError;
								if (lastError) reject(new Error(lastError.message));
								else resolve(resp);
							},
						),
					),
				[`${base}/${file}`, detectorInput],
			);
			const v = res.res;
			const actual: ImageResult["actual"] =
				v?.decision === "blur" || v?.decision === "clear" ? v.decision : "error";
			results.push({
				file,
				split,
				category: entry.category,
				expected: entry.decision,
				actual,
				ms: v?.ms ?? 0,
				detections: (v?.reasons ?? []).map((d) => ({ label: d.label, score: d.score })),
				expectLabels: entry.expect ?? [],
			});
			const mark = actual === entry.decision ? "ok " : "MISS";
			console.log(`${mark} ${split}/${file} → ${actual} (expected ${entry.decision})`);
		}
	} finally {
		await context.close();
		await close();
		await rm(userDataDir, { recursive: true, force: true });
	}

	const engine = arg("--label") ?? "rewrite";
	const out = arg("--out");
	if (out != null) {
		const payload: ResultsFile = { engine, generatedAt: new Date().toISOString(), results };
		await writeFile(out, JSON.stringify(payload, null, 2));
		console.log(`\nwrote ${out}`);
	}

	const baselinePath = arg("--baseline");
	let baseline: ResultsFile | undefined;
	if (baselinePath != null) {
		baseline = JSON.parse(await readFile(baselinePath, "utf8")) as ResultsFile;
		console.log(`baseline: ${baseline.engine} (${baseline.generatedAt})`);
	}

	console.log(`\n## ${engine} — split(s): ${splits.join(", ")}\n`);
	for (const split of splits) {
		const rs = results.filter((r) => r.split === split);
		const rows: [string, Metrics, Metrics?][] = [["ALL", metrics(rs)]];
		const cats = [...new Set(rs.map((r) => r.category))].sort();
		for (const c of cats) {
			const crs = rs.filter((r) => r.category === c);
			const bm = baseline?.results.filter((r) => r.split === split && r.category === c);
			rows.push([c, metrics(crs), bm != null && bm.length > 0 ? metrics(bm) : undefined]);
		}
		printTable(`split: ${split}`, rows);
	}
	printPerClass(results);
	if (baseline != null) {
		const all = metrics(results);
		const ball = metrics(baseline.results.filter((r) => splits.includes(r.split)));
		printTable("head-to-head (selected splits)", [
			[engine, all, ball],
			[baseline.engine, ball, all],
		]);
	}
}

main().catch((err) => {
	console.error(err);
	process.exit(1);
});
