import { expect, type Page, test } from "@playwright/test";
import {
	closeExtension,
	findTabId,
	launchExtension,
	sendProtoMessage,
	serveFixtures,
	writeSettings,
} from "./harness";

/**
 * WP5 integration E2E: the built extension runs the full pipeline on a local
 * fixture page — discovery, fail-closed blur, engine verdicts, video sampling,
 * reveal, allowlist, and the popup's ping/getTabStats round-trips.
 *
 * Requires `pnpm build` + `pnpm models:fetch`; run under `xvfb-run` headless.
 */

const PAGE = `<!doctype html>
<html><head><meta charset="utf-8"><title>hb fixture</title>
<style>
#bg { width:160px; height:120px; background-image:url('/portrait-woman.jpg'); background-size:cover; }
img, video { display:block; margin:4px; }
</style></head><body>
<img id="p-woman" src="/portrait-woman.jpg" width="160" height="120">
<img id="grad" src="/gradient.png" width="160" height="120">
<img id="srcset-img" src="/gradient.png" srcset="/portrait-woman.jpg 480w" sizes="160px" width="160" height="120">
<picture><source srcset="/portrait-woman.jpg"><img id="picture-img" src="/gradient.png" width="160" height="120"></picture>
<div id="bg"></div>
<video id="poster-vid" poster="/portrait-woman.jpg" width="200" height="150"></video>
<canvas id="cv" width="320" height="240" hidden></canvas>
<video id="live-vid" autoplay muted playsinline width="200" height="150"></video>
<iframe id="fr" src="/frame.html" width="300" height="200"></iframe>
<script>
  const cv = document.getElementById('cv');
  const ctx = cv.getContext('2d');
  const im = new Image();
  let offset = 0;
  im.onload = () => {
    const paint = () => {
      ctx.drawImage(im, 0, 0, 320, 240);
      offset = (offset + 6) % 320;
      ctx.fillStyle = '#1144aa';
      // Keep the moving bar under the face region so frames keep changing
      // (no duplicate-skip) without ever occluding the detection target.
      ctx.fillRect(offset, 200, 24, 40);
    };
    paint();
    setInterval(paint, 120);
    const v = document.getElementById('live-vid');
    v.srcObject = cv.captureStream(10);
    v.play().catch(() => {});
  };
  im.src = '/portrait-woman.jpg';
  setTimeout(() => {
    const img = document.createElement('img');
    img.id = 'late';
    img.src = '/portrait-woman.jpg';
    img.width = 160; img.height = 120;
    document.body.appendChild(img);
  }, 1500);
</script>
</body></html>`;

const FRAME = `<!doctype html>
<html><body><img id="frame-img" src="/portrait-woman.jpg" width="120" height="90"></body></html>`;

interface TabStats {
	total: number;
	images: number;
	videos: number;
	byState: Record<string, number>;
}

test("media pipeline: classify, blur, srcset/picture, background, iframe, video", async () => {
	const { context, extensionId, userDataDir } = await launchExtension();
	const { url, close } = await serveFixtures({ "/": PAGE, "/frame.html": FRAME });
	try {
		// strictness 1 lowers policy thresholds — keeps verdicts deterministic
		// for marginal fixture detections and exercises the settings path.
		const ext = await context.newPage();
		await ext.goto(`chrome-extension://${extensionId}/popup.html`);
		await writeSettings(ext, { strictness: 1 });

		const page = await context.newPage();
		await page.goto(url);

		// Fail closed: media is blurred while a verdict is outstanding — this
		// is the cold-start guarantee (engine may still be booting).
		await page.waitForSelector('[data-hb-state="pending"][data-hb-blur]', { timeout: 20_000 });

		// Images classify: portrait-woman blurs, the gradient clears. With
		// regionBlur on (default) the blur is an overlay of detection boxes —
		// data-hb-region, not the whole-element data-hb-blur.
		await expect(page.locator("#p-woman")).toHaveAttribute("data-hb-state", "blur", {
			timeout: 30_000,
		});
		await expect(page.locator("#p-woman")).toHaveAttribute("data-hb-region", "1");
		expect(await page.locator("#p-woman").getAttribute("data-hb-blur")).toBeNull();
		await expect(page.locator("#p-woman")).toHaveAttribute("data-hb-reasons", /FACE_FEMALE/);
		await expect(page.locator("#grad")).toHaveAttribute("data-hb-state", "clear", {
			timeout: 30_000,
		});
		expect(await page.locator("#grad").getAttribute("data-hb-blur")).toBeNull();

		// srcset and <picture> are classified by currentSrc, not src.
		await expect(page.locator("#srcset-img")).toHaveAttribute("data-hb-state", "blur", {
			timeout: 30_000,
		});
		await expect(page.locator("#picture-img")).toHaveAttribute("data-hb-state", "blur", {
			timeout: 30_000,
		});

		// CSS background image — background kind always uses whole-element blur.
		await expect(page.locator("#bg")).toHaveAttribute("data-hb-kind", "background");
		await expect(page.locator("#bg")).toHaveAttribute("data-hb-state", "blur", {
			timeout: 30_000,
		});
		await expect(page.locator("#bg")).toHaveAttribute("data-hb-blur", "1");

		// An image inserted seconds after load is processed (MutationObserver).
		await expect(page.locator("#late")).toHaveAttribute("data-hb-state", "blur", {
			timeout: 30_000,
		});

		// Media inside the iframe is classified by the frame's own script.
		const frame = page.frames().find((f) => f.url().endsWith("/frame.html"));
		expect(frame != null).toBe(true);
		await expect(frame!.locator("#frame-img")).toHaveAttribute("data-hb-state", "blur", {
			timeout: 30_000,
		});

		// Videos: the paused poster video is blurred fail-closed; the live
		// (canvas-streamed) video reaches "blurred" via sampler hysteresis.
		await expect(page.locator("#poster-vid")).toHaveAttribute("data-hb-kind", "video");
		await expect(page.locator("#poster-vid")).toHaveAttribute("data-hb-blur", /.*/, {
			timeout: 30_000,
		});
		// The sampler pauses while a video is off-screen (PLAN.md), so it must
		// be scrolled into the viewport before it can sample.
		await page.locator("#live-vid").scrollIntoViewIfNeeded();
		await expect(page.locator("#live-vid")).toHaveAttribute("data-hb-state", "blurred", {
			timeout: 30_000,
		});
	} finally {
		await closeExtension(context, userDataDir);
		await close();
	}
});

test("reveal on click and hover respect settings", async () => {
	const { context, extensionId, userDataDir } = await launchExtension();
	const { url, close } = await serveFixtures({ "/": PAGE, "/frame.html": FRAME });
	try {
		// Enable reveal before the page loads.
		const ext = await context.newPage();
		await ext.goto(`chrome-extension://${extensionId}/popup.html`);
		await writeSettings(ext, { unblurImages: true, unblurVideos: true });

		const page = await context.newPage();
		await page.goto(url);
		await expect(page.locator("#p-woman")).toHaveAttribute("data-hb-state", "blur", {
			timeout: 30_000,
		});

		// Click reveal: permanent — works from region mode too; the overlay
		// and data-hb-region are removed along with the blur state.
		await page.locator("#p-woman").click();
		await expect(page.locator("#p-woman")).toHaveAttribute("data-hb-reveal", "click");
		expect(await page.locator("#p-woman").getAttribute("data-hb-blur")).toBeNull();
		expect(await page.locator("#p-woman").getAttribute("data-hb-region")).toBeNull();
		expect(await page.locator("#p-woman + .hb-region-overlay").count()).toBe(0);

		// Hover reveal on a region-blurred image fades the overlay out after
		// the 1s delay (the element itself is never CSS-filtered). Wait for
		// the verdict first — hovering a still-pending image finds no overlay.
		await expect(page.locator("#srcset-img")).toHaveAttribute("data-hb-region", "1", {
			timeout: 30_000,
		});
		await page.locator("#srcset-img").hover();
		await page.waitForFunction(
			() => {
				const overlay = document.querySelector(
					"#srcset-img + .hb-region-overlay",
				) as HTMLElement | null;
				if (overlay == null) return false;
				return parseFloat(getComputedStyle(overlay).opacity) < 0.5;
			},
			undefined,
			{ timeout: 15_000, polling: 250 },
		);
	} finally {
		await closeExtension(context, userDataDir);
		await close();
	}
});

test("region blur: overlay tracks the image and toggles with settings", async () => {
	const { context, extensionId, userDataDir } = await launchExtension();
	const { url, close } = await serveFixtures({ "/": PAGE, "/frame.html": FRAME });
	try {
		const ext = await context.newPage();
		await ext.goto(`chrome-extension://${extensionId}/popup.html`);
		await writeSettings(ext, { strictness: 1 });

		const page = await context.newPage();
		await page.goto(url);
		await expect(page.locator("#p-woman")).toHaveAttribute("data-hb-state", "blur", {
			timeout: 30_000,
		});
		await expect(page.locator("#p-woman")).toHaveAttribute("data-hb-region", "1");

		// The overlay is the image's next sibling, covers its box, and holds
		// one backdrop-filter child per blur-capable detection.
		const overlay = page.locator("#p-woman + .hb-region-overlay");
		await expect(overlay).toBeVisible();
		expect(await overlay.locator(".hb-region").count()).toBeGreaterThan(0);
		const imgBox = await page.locator("#p-woman").boundingBox();
		const overlayBox = await overlay.boundingBox();
		expect(imgBox).not.toBeNull();
		expect(overlayBox).not.toBeNull();
		expect(Math.abs(overlayBox!.x - imgBox!.x)).toBeLessThanOrEqual(2);
		expect(Math.abs(overlayBox!.y - imgBox!.y)).toBeLessThanOrEqual(2);
		expect(Math.abs(overlayBox!.width - imgBox!.width)).toBeLessThanOrEqual(2);
		expect(Math.abs(overlayBox!.height - imgBox!.height)).toBeLessThanOrEqual(2);

		// Turning regionBlur off re-decides from stored detections: the same
		// verdict flips to whole-element blur and the overlay is removed.
		await writeSettings(ext, { regionBlur: false });
		await expect(page.locator("#p-woman")).toHaveAttribute("data-hb-blur", "1", {
			timeout: 15_000,
		});
		expect(await page.locator("#p-woman").getAttribute("data-hb-region")).toBeNull();
		expect(await page.locator(".hb-region-overlay").count()).toBe(0);

		// And back on: region mode returns without re-classifying.
		await writeSettings(ext, { regionBlur: true });
		await expect(page.locator("#p-woman")).toHaveAttribute("data-hb-region", "1", {
			timeout: 15_000,
		});
		expect(await page.locator("#p-woman").getAttribute("data-hb-blur")).toBeNull();
		expect(await page.locator("#p-woman + .hb-region-overlay .hb-region").count()).toBeGreaterThan(
			0,
		);
	} finally {
		await closeExtension(context, userDataDir);
		await close();
	}
});

test("allowlisted host is untouched", async () => {
	const { context, extensionId, userDataDir } = await launchExtension();
	const { url, close } = await serveFixtures({ "/": PAGE, "/frame.html": FRAME });
	try {
		const ext = await context.newPage();
		await ext.goto(`chrome-extension://${extensionId}/popup.html`);
		await writeSettings(ext, { allowlist: ["127.0.0.1"] });

		const page = await context.newPage();
		await page.goto(url);
		await page.waitForTimeout(3_000);
		expect(await page.locator("[data-hb-kind]").count()).toBe(0);
	} finally {
		await closeExtension(context, userDataDir);
		await close();
	}
});

test("popup ping, getTabStats, and engineStatus return real values", async () => {
	const { context, extensionId, userDataDir } = await launchExtension();
	const { url, close } = await serveFixtures({ "/": PAGE, "/frame.html": FRAME });
	try {
		const page = await context.newPage();
		await page.goto(url);
		await expect(page.locator("#p-woman")).toHaveAttribute("data-hb-state", "blur", {
			timeout: 30_000,
		});
		await expect(page.locator("#grad")).toHaveAttribute("data-hb-state", "clear", {
			timeout: 30_000,
		});

		const popup: Page = await context.newPage();
		await popup.goto(`chrome-extension://${extensionId}/popup.html`);

		// Protocol ping.
		const pong = await sendProtoMessage<{ pong?: boolean }>(popup, "ping");
		expect(pong.pong).toBe(true);

		// getTabStats reports real counts for the fixture tab.
		const tabId = await findTabId(popup, url);
		expect(tabId != null).toBe(true);
		const stats = await sendProtoMessage<TabStats>(popup, "getTabStats", { tabId });
		expect(stats.total).toBeGreaterThanOrEqual(5);
		expect(stats.images).toBeGreaterThanOrEqual(4);
		expect(stats.videos).toBeGreaterThanOrEqual(2);
		expect((stats.byState.blur ?? 0) + (stats.byState.blurred ?? 0)).toBeGreaterThanOrEqual(4);

		// engineStatus resolves (WP4's "Unavailable" gap is closed) — poll until
		// the offscreen host is up, then confirm the popup renders it. `ready`
		// means WebGPU; the WASM fallback lands in `degraded`, which is the
		// expected steady state in headless CI without a GPU.
		let status = await sendProtoMessage<{ state: string; backend: string | null }>(
			popup,
			"engineStatus",
		);
		for (let i = 0; i < 120 && status.state === "booting"; i++) {
			await popup.waitForTimeout(250);
			status = await sendProtoMessage(popup, "engineStatus");
		}
		expect(["ready", "degraded"]).toContain(status.state);
		expect(status.backend).not.toBeNull();

		await popup.reload();
		const label = status.state === "degraded" ? "Degraded" : "Ready";
		await expect(popup.getByTestId("engine-state")).toContainText(label, {
			timeout: 30_000,
		});
		await expect(popup.getByTestId("media-count")).toBeVisible();
	} finally {
		await closeExtension(context, userDataDir);
		await close();
	}
});
