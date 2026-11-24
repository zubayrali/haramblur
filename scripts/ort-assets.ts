import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";

export interface OrtAsset {
	/** File inside node_modules/onnxruntime-web/dist/. */
	absoluteSrc: string;
	/** Where it lands inside the built extension. */
	relativeDest: string;
}

// onnxruntime-web loads its WASM binaries from `ort.env.wasm.wasmPaths`, which the
// engine sets to browser.runtime.getURL("/ort/") (docs/ENGINE_NOTES.md). Only the
// files the runtime actually fetches are copied: the default simd+threaded build
// and the JSEP variant used by the WebGPU execution provider.
const ORT_DIST_FILES = [
	"ort-wasm-simd-threaded.wasm",
	"ort-wasm-simd-threaded.mjs",
	"ort-wasm-simd-threaded.jsep.wasm",
	"ort-wasm-simd-threaded.jsep.mjs",
] as const;

export function copyOrtAssets(root: string): OrtAsset[] {
	const dist = join(root, "node_modules", "onnxruntime-web", "dist");
	if (!existsSync(dist)) {
		throw new Error(`onnxruntime-web dist not found at ${dist}. Run \`pnpm install\` first.`);
	}
	const present = new Set(readdirSync(dist));
	return ORT_DIST_FILES.map((name) => {
		if (!present.has(name)) {
			throw new Error(`onnxruntime-web is missing expected file ${name}`);
		}
		return {
			absoluteSrc: join(dist, name),
			relativeDest: `ort/${name}`,
		};
	});
}
