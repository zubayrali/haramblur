/**
 * `pnpm models:fetch` — downloads the bundled ML models into git-ignored
 * `public/models/` and verifies pinned SHA-256 hashes (docs/PLAN.md).
 * Skips files that already exist with a valid hash.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, statSync } from "node:fs";
import { readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const MODELS_DIR = join(ROOT, "public", "models");

interface ModelFile {
	url: string;
	fileName: string;
	sha256: string;
	/** Expected size in bytes; checked before hashing for a fast fail. */
	size: number;
}

const MODELS: readonly ModelFile[] = [
	{
		url: "https://huggingface.co/deepghs/nudenet_onnx/resolve/main/320n.onnx",
		fileName: "320n.onnx",
		sha256: "c15d8273adad2d0a92f014cc69ab2d6c311a06777a55545f2c4eb46f51911f0f",
		size: 12_150_158,
	},
	{
		// FastFace small112 (Apache-2.0): MobileNetV3 gender+age on aligned
		// face crops. Re-labels NudeNet's weak FACE_* gender guesses.
		url: "https://huggingface.co/iFurySt/fastface/resolve/main/models/fastface-small-112/model_fp32.onnx",
		fileName: "fastface-small-112.onnx",
		sha256: "0c3981b6c6bc67f32eca6d2d7be70dd3d11aa636c6506b79234adf4bb52d0af5",
		size: 5_114_478,
	},
	{
		// YuNet face detector (Apache-2.0, opencv_zoo): 1-pass letterboxed
		// detector with 5 landmarks per face, strides 8/16/32. Replaces the
		// gated short-range BlazeFace experiment — detections stand alone.
		url: "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
		fileName: "yunet-2023mar.onnx",
		sha256: "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
		size: 232_589,
	},
];

async function sha256OfFile(path: string): Promise<string> {
	return createHash("sha256")
		.update(await readFile(path))
		.digest("hex");
}

async function isValid(path: string, model: ModelFile): Promise<boolean> {
	if (!existsSync(path)) return false;
	if (statSync(path).size !== model.size) return false;
	return (await sha256OfFile(path)) === model.sha256;
}

async function download(url: string, dest: string): Promise<void> {
	const res = await fetch(url, { redirect: "follow" });
	if (!res.ok) throw new Error(`Download failed: ${url} → HTTP ${res.status}`);
	const tmp = `${dest}.part`;
	await writeFile(tmp, Buffer.from(await res.arrayBuffer()));
	await rename(tmp, dest);
}

async function main(): Promise<void> {
	mkdirSync(MODELS_DIR, { recursive: true });
	for (const model of MODELS) {
		const dest = join(MODELS_DIR, model.fileName);
		if (await isValid(dest, model)) {
			console.log(`✓ ${model.fileName} already present and valid`);
			continue;
		}
		await rm(dest, { force: true });
		console.log(`Downloading ${model.fileName} from ${model.url}`);
		await download(model.url, dest);
		const sha = await sha256OfFile(dest);
		if (sha !== model.sha256) {
			await rm(dest, { force: true });
			throw new Error(
				`SHA-256 mismatch for ${model.fileName}: expected ${model.sha256}, got ${sha}`,
			);
		}
		console.log(`✓ ${model.fileName} verified (${model.size} bytes)`);
	}
}

main().catch((err) => {
	console.error(err instanceof Error ? err.message : err);
	process.exit(1);
});
