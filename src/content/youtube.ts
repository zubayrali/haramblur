import type { Settings } from "../shared/settings";

/**
 * YouTube hardening — site-specific shields that don't fit the generic
 * pipeline.
 *
 * Caption cover: captions carry a video's content past the visual censor
 * (a censored clip stays fully readable), so while the player holds a
 * blurred/region-blurred `<video>` its caption overlay is hidden. Pure CSS
 * driven by the pipeline's own `data-hb-*` attributes — no observers, no
 * timers. Self-gating: when the extension is off or the site is allowlisted
 * no video carries those attributes, so the rules never match; revealing a
 * video drops the attributes and its captions return.
 *
 * `.html5-video-player` covers the watch page, embeds and the shorts player
 * alike; the shorts shelf shield lives here too when it lands.
 */

const STYLE_ID = "hb-youtube-shields";

const CAPTION_COVER_CSS = `
.html5-video-player:has(video[data-hb-blur]) .caption-window,
.html5-video-player:has(video[data-hb-blur]) .ytp-caption-window-container,
.html5-video-player:has(video[data-hb-region]) .caption-window,
.html5-video-player:has(video[data-hb-region]) .ytp-caption-window-container {
	display: none !important;
}`;

export function isYouTubeHost(hostname: string): boolean {
	return hostname === "youtube.com" || hostname.endsWith(".youtube.com");
}

/**
 * Inject or remove the shields stylesheet for the current settings.
 * Idempotent; safe at document_start (attaches to documentElement when
 * `<head>` does not exist yet).
 */
export function applyYouTubeShields(
	settings: Settings,
	doc: Document = document,
	hostname: string = location.hostname,
): void {
	const active = isYouTubeHost(hostname) && settings.youtubeCaptionCover;
	let style = doc.getElementById(STYLE_ID) as HTMLStyleElement | null;
	if (!active) {
		style?.remove();
		return;
	}
	if (style == null) {
		style = doc.createElement("style");
		style.id = STYLE_ID;
		(doc.head ?? doc.documentElement).appendChild(style);
	}
	if (style.textContent !== CAPTION_COVER_CSS) style.textContent = CAPTION_COVER_CSS;
}
