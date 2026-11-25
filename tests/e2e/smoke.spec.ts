import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { type BrowserContext, chromium, expect, test } from "@playwright/test";

/** Minimal chrome API surface used inside extension pages during E2E. */
declare const chrome: {
	runtime: {
		sendMessage(message: unknown, callback: (response: { res?: { pong: boolean } }) => void): void;
	};
};

const EXTENSION_DIR = join(
	dirname(fileURLToPath(import.meta.url)),
	"..",
	"..",
	".output",
	"chrome-mv3",
);

/**
 * Smoke test: the unpacked extension must load in Chromium, the background
 * service worker must answer a protocol `ping`, and the popup and options
 * pages must render — including under an RTL UI language.
 *
 * Requires `pnpm build` first. Extensions need a headed browser — run under
 * `xvfb-run` when there is no display (CI does this).
 */
async function launchExtension(lang = "en"): Promise<{
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

test("extension loads and popup/options render", async () => {
	const { context, extensionId, userDataDir } = await launchExtension();
	try {
		const page = await context.newPage();
		await page.goto(`chrome-extension://${extensionId}/popup.html`);
		await expect(page.getByTestId("popup-root")).toBeVisible();
		await expect(page.getByTestId("master-toggle")).toBeVisible();
		await expect(page.getByTestId("engine-state")).toBeVisible();

		// Protocol round-trip: the background answers `ping` via
		// @webext-core/messaging's wire format.
		const pong = await page.evaluate(
			() =>
				new Promise<{ res?: { pong: boolean } }>((resolve) => {
					chrome.runtime.sendMessage(
						{ id: 1, type: "ping", data: undefined, timestamp: Date.now() },
						resolve,
					);
				}),
		);
		expect(pong.res?.pong).toBe(true);

		await page.goto(`chrome-extension://${extensionId}/options.html`);
		await expect(page.getByTestId("options-root")).toBeVisible();
		// The General tab is the default panel.
		await expect(page.getByTestId("allowlist-editor")).toBeVisible();
		await expect(page.getByTestId("blur-preview")).not.toBeAttached();

		// Tabs switch panels and deep-link through the URL hash.
		await page.getByTestId("tab-display").click();
		await expect(page.getByTestId("blur-preview")).toBeVisible();
		expect(page.url()).toContain("#display");

		await page.getByTestId("tab-models").click();
		await expect(page.getByTestId("section-models")).toBeVisible();
		await expect(page.getByTestId("blur-preview")).not.toBeAttached();

		// A hash in the URL pre-selects the tab on load.
		await page.goto(`chrome-extension://${extensionId}/options.html#display`);
		await expect(page.getByTestId("blur-preview")).toBeVisible();
	} finally {
		await context.close();
		await rm(userDataDir, { recursive: true, force: true });
	}
});

test("popup and options render RTL in Arabic", async () => {
	const { context, extensionId, userDataDir } = await launchExtension("ar");
	try {
		const page = await context.newPage();
		await page.goto(`chrome-extension://${extensionId}/popup.html`);
		await expect(page.getByTestId("popup-root")).toBeVisible();
		expect(await page.evaluate(() => document.documentElement.dir)).toBe("rtl");
		await expect(page.getByTestId("master-toggle")).toHaveAttribute("aria-label", "تفعيل الحماية");

		await page.goto(`chrome-extension://${extensionId}/options.html`);
		await expect(page.getByTestId("options-root")).toBeVisible();
		expect(await page.evaluate(() => document.documentElement.dir)).toBe("rtl");
	} finally {
		await context.close();
		await rm(userDataDir, { recursive: true, force: true });
	}
});
