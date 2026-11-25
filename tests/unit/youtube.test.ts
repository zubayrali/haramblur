// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { applyYouTubeShields, isYouTubeHost } from "../../src/content/youtube";
import { DEFAULT_SETTINGS } from "../../src/shared/settings";

describe("isYouTubeHost", () => {
	it("matches youtube.com and its subdomains only", () => {
		expect(isYouTubeHost("youtube.com")).toBe(true);
		expect(isYouTubeHost("www.youtube.com")).toBe(true);
		expect(isYouTubeHost("m.youtube.com")).toBe(true);
		expect(isYouTubeHost("notyoutube.com")).toBe(false);
		expect(isYouTubeHost("youtube.com.evil.example")).toBe(false);
		expect(isYouTubeHost("vimeo.com")).toBe(false);
	});
});

describe("applyYouTubeShields", () => {
	function freshDocument(): Document {
		// jsdom always has <head>, so the stylesheet never lands on documentElement here.
		document.head.innerHTML = "";
		return document;
	}

	it("injects the caption-cover stylesheet on YouTube when the setting is on", () => {
		const doc = freshDocument();
		applyYouTubeShields({ ...DEFAULT_SETTINGS, youtubeCaptionCover: true }, doc, "www.youtube.com");
		const style = doc.getElementById("hb-youtube-shields");
		expect(style).not.toBeNull();
		expect(style?.textContent).toContain(".html5-video-player:has(video[data-hb-blur])");
		expect(style?.textContent).toContain("[data-hb-region]");
		expect(style?.textContent).toContain(".caption-window");
	});

	it("injects nothing off YouTube", () => {
		const doc = freshDocument();
		applyYouTubeShields({ ...DEFAULT_SETTINGS, youtubeCaptionCover: true }, doc, "example.com");
		expect(doc.getElementById("hb-youtube-shields")).toBeNull();
	});

	it("removes the stylesheet when the setting turns off", () => {
		const doc = freshDocument();
		applyYouTubeShields({ ...DEFAULT_SETTINGS, youtubeCaptionCover: true }, doc, "www.youtube.com");
		expect(doc.getElementById("hb-youtube-shields")).not.toBeNull();
		applyYouTubeShields(
			{ ...DEFAULT_SETTINGS, youtubeCaptionCover: false },
			doc,
			"www.youtube.com",
		);
		expect(doc.getElementById("hb-youtube-shields")).toBeNull();
	});

	it("is idempotent", () => {
		const doc = freshDocument();
		applyYouTubeShields({ ...DEFAULT_SETTINGS, youtubeCaptionCover: true }, doc, "www.youtube.com");
		applyYouTubeShields({ ...DEFAULT_SETTINGS, youtubeCaptionCover: true }, doc, "www.youtube.com");
		expect(doc.querySelectorAll("#hb-youtube-shields").length).toBe(1);
	});
});
