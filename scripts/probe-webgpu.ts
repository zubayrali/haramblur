/**
 * `pnpm probe:webgpu` — DP0 measurement spike. Loads the built Chrome MV3
 * extension, asks the real offscreen-document engine host for adapter/limits
 * plus fresh-session zero-tensor timings, and prints a table suitable for the
 * DP0 PR description. Requires `pnpm build` first.
 */
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import type { WebGpuProbeReport } from "../src/shared/webgpu-probe";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const EXTENSION_DIR = join(ROOT, ".output", "chrome-mv3");
const DEFAULT_JSON_OUT = join(ROOT, "test-results", "webgpu-probe.json");

interface MessageResponse {
	res?: unknown;
	err?: unknown;
}

declare const chrome: {
	runtime: {
		sendMessage(message: unknown, callback: (response: MessageResponse) => void): void;
		lastError?: { message?: string };
	};
};

const fmt = (value: number | undefined, suffix = "") =>
	value == null ? "—" : `${value.toFixed(value >= 100 ? 0 : 2)}${suffix}`;
const yn = (value: boolean) => (value ? "yes" : "no");

async function main(): Promise<void> {
	if (!existsSync(EXTENSION_DIR)) {
		throw new Error(`${EXTENSION_DIR} missing — run \`pnpm build\` first`);
	}
	const jsonOut = process.argv.includes("--no-json") ? null : DEFAULT_JSON_OUT;
	const userDataDir = await mkdtemp(join(tmpdir(), "hb-webgpu-probe-"));
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
		const client = await page.evaluate(() => ({
			userAgent: navigator.userAgent,
			hardwareConcurrency: navigator.hardwareConcurrency,
		}));
		console.log(`browser: ${client.userAgent}; hardwareConcurrency=${client.hardwareConcurrency}`);

		const call = async <T>(type: string, data?: unknown): Promise<T> => {
			const env = await page.evaluate(
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
			);
			if (env.err != null) throw new Error(`protocol error: ${JSON.stringify(env.err)}`);
			return env.res as T;
		};

		const started = Date.now();
		let status: { state: string; backend: string | null; error?: string };
		do {
			status = await call("engineStatus");
			await new Promise((resolve) => setTimeout(resolve, 50));
		} while (status.state === "booting" && Date.now() - started < 60_000);
		const loadMs = Date.now() - started;
		console.log(`engine: state=${status.state} backend=${status.backend} load=${loadMs}ms`);
		if (status.state !== "ready" && status.state !== "degraded") {
			throw new Error(`engine never became ready: ${status.error ?? status.state}`);
		}

		const report = await call<WebGpuProbeReport>("webgpuProbe");
		if (report.error != null) console.log(`probe transport error: ${report.error}`);
		console.log(
			"\n| context | navigator.gpu | adapter | fp16 | maxBufferSize | maxComputeWorkgroups | session create | cold zero run | warm mean | warm min | outcome |",
		);
		console.log("|---|---|---|---|---|---|---|---|---|---|---|");
		for (const ctx of report.contexts) {
			const z = ctx.zeroTensor;
			const outcome = z.error ?? (z.attempted ? "ok" : "not attempted");
			console.log(
				`| ${ctx.context} | ${yn(ctx.adapter.navigatorGpu)} | ${yn(ctx.adapter.adapter)} | ${yn(
					ctx.adapter.shaderF16,
				)} | ${fmt(ctx.adapter.limits.maxBufferSize)} | ${fmt(
					ctx.adapter.limits.maxComputeWorkgroupsPerDimension,
				)} | ${fmt(z.sessionCreateMs, "ms")} | ${fmt(z.coldRunMs, "ms")} | ${fmt(
					z.warmMeanMs,
					"ms",
				)} | ${fmt(z.warmMinMs, "ms")} | ${outcome} |`,
			);
		}
		for (const ctx of report.contexts) {
			const info = ctx.adapter.info;
			console.log(
				`\n${ctx.context}: features=[${ctx.adapter.features.join(", ") || "none"}] ` +
					`fallback=${ctx.adapter.isFallbackAdapter ?? "unknown"} ` +
					`info=${JSON.stringify(info ?? {})}`,
			);
			if (ctx.zeroTensor.outputDims != null) {
				console.log(`${ctx.context}: outputDims=${JSON.stringify(ctx.zeroTensor.outputDims)}`);
			}
		}
		if (jsonOut != null) {
			await mkdir(dirname(jsonOut), { recursive: true });
			await writeFile(
				jsonOut,
				`${JSON.stringify({ client, status, loadMs, report, measuredAt: new Date().toISOString() }, null, 2)}\n`,
			);
			console.log(`\njson: ${jsonOut}`);
		}
	} finally {
		await context.close();
		await rm(userDataDir, { recursive: true, force: true });
	}
}

main().catch((err) => {
	console.error(err instanceof Error ? err.message : err);
	process.exit(1);
});
