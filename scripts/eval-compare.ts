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

function printTable(title: string, rows: [string, Metrics, Metrics?][]): void {
	console.log(`\n### ${title}\n`);
	console.log(
		"| set | n | acc | prec | rec | F1 | false-blur | miss | p50ms | p95ms |" +
			(rows.some((r) => r[2]) ? " vs-base acc/F1 |" : ""),
	);
	console.log("|---|---|---|---|---|---|---|---|---|---|" + (rows.some((r) => r[2]) ? "---|" : ""));
	for (const [name, m, base] of rows) {
		const line = `| ${name} | ${m.n} | ${pct(m.acc)} | ${pct(m.precision)} | ${pct(m.recall)} | ${pct(m.f1)} | ${pct(m.falseBlur)} | ${pct(m.miss)} | ${m.p50} | ${m.p95} |`;
		console.log(base ? `${line} ${pct(base.acc)}/${pct(base.f1)} |` : line);
	}
}

async function main(): Promise<void> {
	if (!existsSync(EXTENSION_DIR)) throw new Error(`${EXTENSION_DIR} missing — run pnpm build`);
	const manifest = JSON.parse(await readFile(MANIFEST_V2, "utf8")) as ManifestV2;
	const splitArg = arg("--split") ?? "test";
	const splits =
		splitArg === "all" ? Object.keys(manifest.splits) : [splitArg];
	const entries: { file: string; split: string; entry: ManifestEntry }[] = [];
	for (const s of splits) {
		const files = manifest.splits[s]?.files;
		if (files == null) throw new Error(`unknown split "${s}"`);
		for (const [file, entry] of Object.entries(files)) entries.push({ file, split: s, entry });
	}

	const { url: base, close } = await serveFixtures();
	const userDataDir = await mkdtemp(join(tmpdir(), "hb-cmp-"));
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

		for (const { file, split, entry } of entries) {
			const res = await page.evaluate(
				(url) =>
					new Promise<WireResponse>((resolve, reject) =>
						chrome.runtime.sendMessage(
							{ id: 1, type: "classifyImage", data: { url }, timestamp: Date.now() },
							(resp: WireResponse) => {
								const lastError = chrome.runtime.lastError;
								if (lastError) reject(new Error(lastError.message));
								else resolve(resp);
							},
						),
					),
				`${base}/${file}`,
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
