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
				? ["storage", "contextMenus"]
				: ["storage", "offscreen", "contextMenus"],
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
	},
});
