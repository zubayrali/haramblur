/**
 * Canvas capture for media the engine cannot fetch itself (docs/PLAN.md
 * "Image pixels"): `data:`/`blob:` URLs, and `<video>` elements without a
 * fetchable `poster`. Produces a downscaled PNG data URL sent as
 * `classifyImage`'s `frameUrl` (PNG, not JPEG: lossy re-encoding collapses
 * detection scores on these small frames into false `clear` verdicts).
 */

/** Largest edge of the captured frame; matches the model's input size. */
export const FRAME_MAX_EDGE = 320;

const VIDEO_WAIT_MS = 5_000;

function drawScaled(
	source: HTMLImageElement | HTMLVideoElement,
	sw: number,
	sh: number,
): string | null {
	if (sw <= 0 || sh <= 0) return null;
	const scale = Math.min(1, FRAME_MAX_EDGE / Math.max(sw, sh));
	const w = Math.max(1, Math.round(sw * scale));
	const h = Math.max(1, Math.round(sh * scale));
	const canvas = document.createElement("canvas");
	canvas.width = w;
	canvas.height = h;
	const ctx = canvas.getContext("2d");
	if (ctx == null) return null;
	try {
		ctx.drawImage(source, 0, 0, w, h);
		return canvas.toDataURL("image/png");
	} catch {
		// Cross-origin media without CORS taints the canvas.
		return null;
	}
}

/**
 * Capture the currently displayed pixels of an image or video as a PNG data
 * URL, at most 320 px on the long edge. Returns `null` when pixels are not
 * drawable (unloaded, or canvas tainted by CORS).
 */
export async function captureMediaFrame(
	el: HTMLImageElement | HTMLVideoElement,
): Promise<string | null> {
	if (el instanceof HTMLImageElement) {
		if (el.naturalWidth === 0 || el.naturalHeight === 0) {
			if (!el.complete) await once(el, ["load", "error"], VIDEO_WAIT_MS);
			if (el.naturalWidth === 0 || el.naturalHeight === 0) return null;
		}
		return drawScaled(el, el.naturalWidth, el.naturalHeight);
	}
	// Video: wait for the first frame if metadata isn't there yet.
	if (el.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
		await once(el, ["loadeddata", "error"], VIDEO_WAIT_MS);
	}
	if (el.videoWidth === 0 || el.videoHeight === 0) return null;
	return drawScaled(el, el.videoWidth, el.videoHeight);
}

/** Wait for one of `events` on `el`, bounded by `timeoutMs`. */
function once(el: HTMLElement, events: string[], timeoutMs: number): Promise<void> {
	return new Promise((resolve) => {
		const timer = setTimeout(done, timeoutMs);
		function done() {
			clearTimeout(timer);
			for (const name of events) el.removeEventListener(name, done);
			resolve();
		}
		for (const name of events) el.addEventListener(name, done, { once: true });
	});
}
