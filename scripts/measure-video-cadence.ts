/**
 * `pnpm measure:video` — measure effective video re-detection cadence in the
 * real built extension. The fixtures stream continuously changing canvases
 * into `<video>` elements; every sampler verdict rewrites a small state
 * attribute, so a MutationObserver gives a close external measure of completed
 * capture→classify chains without instrumenting production code.
 *
 * `clear-motion` is the strict DP2 motion case: verdicts stay clear, so only
 * real motion can keep backoff reset. `blur-motion` additionally exercises
 * the region-update path. Measurement-only by default because the ≥6 fps
 * target is specifically the GPU path; set HB_VIDEO_EXPECT_FPS=6 to gate.
 */

import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { BrowserContext } from "@playwright/test";
import {
	closeExtension,
	EXTENSION_DIR,
	launchExtension,
	sendProtoMessage,
	serveFixtures,
	writeSettings,
} from "../tests/e2e/harness";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = join(ROOT, "test-results", "video-cadence.json");
const MEASURE_MS = Number(process.env.HB_VIDEO_MEASURE_MS ?? 5_000);
const EXPECT_FPS = process.env.HB_VIDEO_EXPECT_FPS ? Number(process.env.HB_VIDEO_EXPECT_FPS) : null;

const VIDEO_OPEN = `<!doctype html>
<html><head><meta charset="utf-8"><title>video cadence fixture</title></head><body>
<canvas id="cv" width="320" height="240" hidden></canvas>
<video id="live-vid" autoplay muted playsinline width="320" height="240"></video>
<script>
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
let frame = 0;
function start(paint) {
  paint();
  setInterval(paint, 33);
  const v = document.getElementById('live-vid');
  v.srcObject = cv.captureStream(30);
  v.play().catch(() => {});
}
`;

const CLEAR_PAGE = `${VIDEO_OPEN}
// Full-height moving stripes: unlike a translating solid block, most luma
// hash cells change on every shift, so this is unambiguous active motion.
start(() => {
  frame++;
  ctx.fillStyle = '#263238';
  ctx.fillRect(0, 0, 320, 240);
  const offset = (frame * 7) % 48;
  for (let x = -48 + offset; x < 340; x += 48) {
    ctx.fillStyle = '#d8d8d8';
    ctx.fillRect(x, 0, 24, 240);
    ctx.fillStyle = '#806000';
    ctx.fillRect(x + 24, 0, 12, 240);
  }
});
</script>
</body></html>`;

const BLUR_PAGE = `${VIDEO_OPEN}
const im = new Image();
im.onload = () => start(() => {
  frame++;
  ctx.drawImage(im, 0, 0, 320, 240);
  // Keep the changing region below the face used by the detector fixture.
  const offset = (frame * 9) % 48;
  for (let x = -48 + offset; x < 340; x += 48) {
    ctx.fillStyle = '#f4f4f4';
    ctx.fillRect(x, 178, 24, 62);
    ctx.fillStyle = '#0055ff';
    ctx.fillRect(x + 24, 178, 12, 62);
  }
});
im.src = '/portrait-woman.jpg';
</script>
</body></html>`;

interface EngineStatusLike {
	state?: string;
	backend?: string | null;
	backends?: Record<string, string>;
	loadMs?: number;
	error?: string;
}

interface ModeSpec {
	name: string;
	path: string;
	expectedState: "clear" | "blur";
	attribute: "data-hb-state" | "data-hb-reasons";
}

interface ModeMeasurement {
	verdictEvents: number;
	firstLastSpanMs: number;
	meanIntervalMs: number;
	effectiveFps: number;
	playbackSeconds: number;
	finalState: string | null;
}

const MODES: ModeSpec[] = [
	{
		name: "clear-motion",
		path: "/clear",
		expectedState: "clear",
		attribute: "data-hb-state",
	},
	{
		name: "blur-motion",
		path: "/blur",
		expectedState: "blur",
		attribute: "data-hb-reasons",
	},
];

async function measureMode(
	context: BrowserContext,
	baseUrl: string,
	mode: ModeSpec,
	measureMs: number,
): Promise<ModeMeasurement> {
	const page = await context.newPage();
	try {
		await page.goto(`${baseUrl}${mode.path}`);
		await page.locator("#live-vid").scrollIntoViewIfNeeded();
		await page.waitForFunction(
			(expected) => document.getElementById("live-vid")?.getAttribute("data-hb-state") === expected,
			mode.expectedState,
			{ timeout: 30_000, polling: 100 },
		);

		const result = await page.evaluate(
			async ({ measureMs: duration, attribute }) => {
				const video = document.getElementById("live-vid") as HTMLVideoElement | null;
				if (video == null) throw new Error("live-vid missing");
				const times: number[] = [];
				const observer = new MutationObserver((records) => {
					const now = performance.now();
					for (const record of records) {
						if (record.attributeName === attribute) times.push(now);
					}
				});
				observer.observe(video, { attributes: true, attributeFilter: [attribute] });
				const startTime = video.currentTime;
				const startPerf = performance.now();
				await new Promise((resolve) => setTimeout(resolve, duration));
				observer.disconnect();
				return {
					times,
					playbackSeconds: video.currentTime - startTime,
					wallMs: performance.now() - startPerf,
					finalState: video.getAttribute("data-hb-state"),
				};
			},
			{ measureMs, attribute: mode.attribute },
		);

		const times = result.times;
		if (times.length < 2) {
			throw new Error(
				`${mode.name}: only ${times.length} verdict mutations observed in ${result.wallMs}ms`,
			);
		}
		const spanMs = times.at(-1)! - times[0]!;
		const intervals = times.slice(1).map((t, i) => t - times[i]!);
		const meanIntervalMs = intervals.reduce((a, b) => a + b, 0) / intervals.length;
		return {
			verdictEvents: times.length,
			firstLastSpanMs: spanMs,
			meanIntervalMs,
			effectiveFps: spanMs > 0 ? ((times.length - 1) * 1000) / spanMs : 0,
			playbackSeconds: result.playbackSeconds,
			finalState: result.finalState,
		};
	} finally {
		await page.close();
	}
}

async function main(): Promise<void> {
	if (!existsSync(EXTENSION_DIR)) {
		throw new Error(`${EXTENSION_DIR} missing — run \`pnpm build\` first`);
	}
	const { context, extensionId, userDataDir } = await launchExtension();
	const { url, close } = await serveFixtures({ "/clear": CLEAR_PAGE, "/blur": BLUR_PAGE });
	try {
		const ext = await context.newPage();
		await ext.goto(`chrome-extension://${extensionId}/popup.html`);
		// Make the portrait fixture deterministically blur.
		await writeSettings(ext, { strictness: 1 });

		const modes: Record<string, ModeMeasurement> = {};
		for (const mode of MODES) {
			const m = await measureMode(context, url, mode, MEASURE_MS);
			modes[mode.name] = m;
			console.log(
				`${mode.name}: events=${m.verdictEvents} fps=${m.effectiveFps.toFixed(2)} ` +
					`meanInterval=${m.meanIntervalMs.toFixed(1)}ms playback=${m.playbackSeconds.toFixed(2)}s ` +
					`finalState=${m.finalState}`,
			);
		}

		const status = await sendProtoMessage<EngineStatusLike>(ext, "engineStatus");
		const report = {
			measuredAt: new Date().toISOString(),
			measureMs: MEASURE_MS,
			modes,
			engine: status,
		};
		await mkdir(dirname(OUT), { recursive: true });
		await writeFile(OUT, `${JSON.stringify(report, null, 2)}\n`);
		console.log(`engine: ${JSON.stringify(status)}`);
		console.log(`wrote ${OUT}`);
		if (EXPECT_FPS != null) {
			for (const [name, m] of Object.entries(modes)) {
				if (m.effectiveFps < EXPECT_FPS) {
					throw new Error(
						`${name} effective fps ${m.effectiveFps.toFixed(2)} below required ${EXPECT_FPS}`,
					);
				}
			}
		}
	} finally {
		await closeExtension(context, userDataDir);
		await close();
	}
}

main().catch((err) => {
	console.error(err instanceof Error ? err.message : err);
	process.exit(1);
});
