import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, expect, test } from "@playwright/test";

const EXTENSION_DIR = join(
	dirname(fileURLToPath(import.meta.url)),
	"..",
	"..",
	".output",
	"chrome-mv3",
);

/**
 * Smoke test: the unpacked extension must load in Chromium and the background
 * service worker must answer a protocol `ping`.
 *
 * Requires `pnpm build` first. Extensions need a headed browser — run under
 * `xvfb-run` when there is no display (CI does this).
 */
test("extension loads and background answers ping", async () => {
	const userDataDir = await mkdtemp(join(tmpdir(), "hb-e2e-"));
	const context = await chromium.launchPersistentContext(userDataDir, {
		headless: false,
		args: [`--disable-extensions-except=${EXTENSION_DIR}`, `--load-extension=${EXTENSION_DIR}`],
	});
	try {
		let worker = context.serviceWorkers().find((sw) => sw.url().startsWith("chrome-extension://"));
		worker ??= await context.waitForEvent("serviceworker", {
			timeout: 15_000,
		});
		expect(worker.url()).toContain("chrome-extension://");
		const extensionId = new URL(worker.url()).host;

		const page = await context.newPage();
		await page.goto(`chrome-extension://${extensionId}/popup.html`);
		await expect(page.getByTestId("ping")).toBeVisible();
		await page.getByTestId("ping").click();
		await expect(page.getByTestId("pong")).toContainText("pong", {
			timeout: 10_000,
		});
	} finally {
		await context.close();
		await rm(userDataDir, { recursive: true, force: true });
	}
});
