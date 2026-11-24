import { existsSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "wxt";
import { copyOrtAssets } from "./scripts/ort-assets";

export default defineConfig({
	srcDir: "src",
	manifest: ({ browser }) => ({
		name: "__MSG_extName__",
		description: "__MSG_extDescription__",
		default_locale: "en",
		// Firefox has no offscreen API — the engine runs in the background page
		// there instead (docs/PLAN.md "Engine host").
		permissions:
			browser === "firefox"
				? ["storage", "contextMenus", "scripting"]
				: ["storage", "offscreen", "contextMenus", "scripting"],
		// WASM instantiation needs 'wasm-unsafe-eval' under the MV3 extension-page
		// CSP ("script-src 'self'" alone rejects WebAssembly.instantiate). onnxruntime
		// compiles its kernels inside the engine host document. Chrome only — Firefox
		// MV3 does not accept an extension_pages CSP object.
		...(browser === "firefox"
			? {}
			: {
					content_security_policy: {
						extension_pages: "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'",
					},
				}),
		host_permissions: ["*://*/*"],
		icons: {
			"48": "/icons/hb-icon-48.png",
			"128": "/icons/hb-icon-128.png",
		},
		browser_specific_settings: {
			gecko: {
				id: "haramblur@zubayrali.github.io",
				strict_min_version: "109.0",
			},
		},
	}),
	vite: () => ({
		plugins: [svelte(), tailwindcss()],
	}),
	hooks: {
		// Ship the onnxruntime-web WASM runtime inside the extension: the engine
		// worker resolves `browser.runtime.getURL("/ort/")` (see docs/ENGINE_NOTES.md).
		"build:publicAssets": (wxt, files) => {
			files.push(...copyOrtAssets(wxt.config.root));
		},
		// Vite emits `<script type="module" crossorigin>` on entry pages. Inside
		// Chrome offscreen documents the CORS-mode fetch of a chrome-extension://
		// module fails, so the host module never executes; dropping the attribute
		// restores a plain same-origin fetch.
		"build:done": async (wxt) => {
			const htmlPath = join(wxt.config.outDir, "offscreen.html");
			if (!existsSync(htmlPath)) return;
			const html = await readFile(htmlPath, "utf8");
			await writeFile(htmlPath, html.replace(/ crossorigin/g, ""));
		},
	},
});
