import type { Canvas2dLike, CanvasLike, SamplerEnvironment } from "./types";

/** Long-edge cap for frames sent to the engine (docs/PLAN.md). */
export const MAX_FRAME_EDGE = 480;
/**
 * Frame payload encoding: WebP q0.92 — ~3× smaller than PNG at this size with
 * quality the detector tolerates (unlike JPEG, whose blocking artifacts
 * collapse detection scores on small frames: a ~0.44 face on the source
 * scored ~0.16 after JPEG q0.95, producing false `clear` verdicts). PNG is
 * the fallback for encoders without WebP.
 */
export const FRAME_MIME = "image/webp";
export const FRAME_QUALITY = 0.92;
export const FRAME_MIME_FALLBACK = "image/png";
/** Per-cell luma hash is HASH_GRID × HASH_GRID. */
export const HASH_GRID = 16;

export interface CapturedFrame {
	dataUrl: string;
	width: number;
	height: number;
	/** Per-cell luma (0..255) on a HASH_GRID² grid — cheap pixel hash. */
	hash: Uint8Array;
	/** `video.currentTime` when the frame was captured. */
	ts: number;
}

/** Fit `w`/`h` inside a `maxEdge` box, preserving aspect ratio. */
export function scaledSize(
	w: number,
	h: number,
	maxEdge = MAX_FRAME_EDGE,
): { width: number; height: number } {
	const scale = Math.min(1, maxEdge / Math.max(w, h));
	return {
		width: Math.max(1, Math.round(w * scale)),
		height: Math.max(1, Math.round(h * scale)),
	};
}

/** Mean absolute per-cell difference of two hashes, normalized to 0..1. */
export function hashDistance(a: Uint8Array, b: Uint8Array): number {
	if (a.length !== b.length || a.length === 0) return 1;
	let sum = 0;
	for (let i = 0; i < a.length; i++) {
		sum += Math.abs((a[i] ?? 0) - (b[i] ?? 0));
	}
	return sum / (a.length * 255);
}

function lumaHash(ctx: Canvas2dLike, width: number, height: number): Uint8Array {
	const img = ctx.getImageData(0, 0, width, height).data;
	const hash = new Uint8Array(HASH_GRID * HASH_GRID);
	const cw = width / HASH_GRID;
	const ch = height / HASH_GRID;
	for (let gy = 0; gy < HASH_GRID; gy++) {
		for (let gx = 0; gx < HASH_GRID; gx++) {
			const x = Math.floor((gx + 0.5) * cw);
			const y = Math.floor((gy + 0.5) * ch);
			const i = (y * width + x) * 4;
			// ITU-R BT.601 luma.
			const r = img[i] ?? 0;
			const g = img[i + 1] ?? 0;
			const b = img[i + 2] ?? 0;
			hash[gy * HASH_GRID + gx] = (299 * r + 587 * g + 114 * b) / 1000;
		}
	}
	return hash;
}

/**
 * Encode the canvas as a frame data URL. WebP is preferred; encoders that
 * don't support it silently produce PNG (convertToBlob) or an empty/PNG
 * data URL (toDataURL), both of which fall back cleanly.
 */
async function canvasToFrameDataUrl(canvas: CanvasLike): Promise<string> {
	if ("convertToBlob" in canvas) {
		let blob = await canvas.convertToBlob({ type: FRAME_MIME, quality: FRAME_QUALITY });
		if (blob.type !== FRAME_MIME) {
			blob = await canvas.convertToBlob({ type: FRAME_MIME_FALLBACK });
		}
		const bytes = new Uint8Array(await blob.arrayBuffer());
		let bin = "";
		for (let i = 0; i < bytes.length; i += 0x8000) {
			bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
		}
		return `data:${blob.type};base64,${btoa(bin)}`;
	}
	const url = (canvas as HTMLCanvasElement).toDataURL(FRAME_MIME, FRAME_QUALITY);
	if (url.startsWith("data:image/webp")) return url;
	return (canvas as HTMLCanvasElement).toDataURL(FRAME_MIME_FALLBACK);
}

/**
 * Draw the video's current frame scaled to at most `MAX_FRAME_EDGE` on the
 * long edge and return it as a WebP (PNG-fallback) data URL plus a cheap
 * luma hash used for duplicate detection and scene-change detection.
 *
 * Returns `null` when the element has no drawable frame yet.
 */
export async function captureFrame(
	video: HTMLVideoElement,
	env: SamplerEnvironment,
): Promise<CapturedFrame | null> {
	if (video.readyState < 2 /* HAVE_CURRENT_DATA */) return null;
	const vw = video.videoWidth;
	const vh = video.videoHeight;
	if (!vw || !vh) return null;

	const { width, height } = scaledSize(vw, vh);
	const canvas = env.createCanvas(width, height);
	const ctx = canvas.getContext("2d") as Canvas2dLike | null;
	if (!ctx) return null;
	ctx.drawImage(video, 0, 0, width, height);
	const dataUrl = await canvasToFrameDataUrl(canvas);
	return {
		dataUrl,
		width,
		height,
		hash: lumaHash(ctx, width, height),
		ts: video.currentTime,
	};
}
