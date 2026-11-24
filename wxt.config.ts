import { existsSync, readdirSync } from "node:fs";
import { readFile, rm, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
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
				// Module Workers (the engine's `new Worker(url, { type: "module" })`)
				// ship in Firefox 114; OffscreenCanvas in workers is covered there too.
				strict_min_version: "114.0",
			},
		},
	}),
	vite: () => ({
		// Use onnxruntime-web's extern-wasm build: the default `ort.bundle.min.mjs`
		// inlines the ~30 MB jsep wasm binary as base64 twice (a ~79 MB worker
		// bundle); the extern build resolves wasm through `wasmPaths` (`/ort/`).
		// An alias is used rather than a resolve condition — `resolve.conditions`
		// replaces Vite's defaults, dropping "browser" and turning svelte into its
		// server build (no `mount`).
		resolve: {
			alias: {
				"onnxruntime-web": join(
					dirname(createRequire(import.meta.url).resolve("onnxruntime-web")),
					"ort.min.mjs",
				),
			},
		},
		plugins: [svelte(), tailwindcss()],
	}),
	hooks: {
		// Ship the onnxruntime-web WASM runtime inside the extension: the engine
		// worker resolves `browser.runtime.getURL("/ort/")` (see docs/ENGINE_NOTES.md).
		"build:publicAssets": (wxt, files) => {
			files.push(...copyOrtAssets(wxt.config.root));
		},
		// Firefox has no offscreen API — the engine runs in the background page
		// there — so the Chrome-only offscreen host page isn't built for it.
		"entrypoints:resolved": (wxt, entrypoints) => {
			if (wxt.config.browser !== "firefox") return;
			for (const entrypoint of entrypoints) {
				if (entrypoint.name === "offscreen") entrypoint.skipped = true;
			}
		},
		// Vite emits `<script type="module" crossorigin>` on entry pages. Inside
		// Chrome offscreen documents the CORS-mode fetch of a chrome-extension://
		// module fails, so the host module never executes; dropping the attribute
		// restores a plain same-origin fetch.
		"build:done": async (wxt) => {
			const htmlPath = join(wxt.config.outDir, "offscreen.html");
			if (existsSync(htmlPath)) {
				const html = await readFile(htmlPath, "utf8");
				await writeFile(htmlPath, html.replace(/ crossorigin/g, ""));
			}
			// Vite emits a hashed copy of ort's wasm binaries under `assets/`, but
			// the runtime only ever fetches them through `wasmPaths` (`/ort/`).
			// The bundled copies are dead weight (~30 MB) — drop them.
			const assetsDir = join(wxt.config.outDir, "assets");
			if (!existsSync(assetsDir)) return;
			for (const file of readdirSync(assetsDir)) {
				if (/^ort-wasm-.+\.wasm$/.test(file)) await rm(join(assetsDir, file));
			}
		},
	},
});
