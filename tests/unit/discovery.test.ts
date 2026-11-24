// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";
import {
	candidateFor,
	extractBackgroundImageUrl,
	imageUrl,
	isClassifiableUrl,
	isTrackingPixel,
	mediaElements,
	videoUrl,
} from "../../src/content/discovery";

const BIG_DATA_URL = `data:image/png;base64,${"A".repeat(2048)}`;

describe("isClassifiableUrl", () => {
	it("accepts ordinary http(s) image URLs", () => {
		expect(isClassifiableUrl("https://cdn.example.com/photo.jpg")).toBe(true);
	});

	it("rejects empty and missing URLs", () => {
		expect(isClassifiableUrl(null)).toBe(false);
		expect(isClassifiableUrl("")).toBe(false);
		expect(isClassifiableUrl(undefined)).toBe(false);
	});

	it("rejects data: URLs under ~1 KB", () => {
		expect(isClassifiableUrl("data:image/png;base64,AAAA")).toBe(false);
		expect(isClassifiableUrl(BIG_DATA_URL)).toBe(true);
	});

	it("rejects svg files and svg data: URLs", () => {
		expect(isClassifiableUrl("https://example.com/icon.svg")).toBe(false);
		expect(isClassifiableUrl(`data:image/svg+xml;base64,${"A".repeat(2048)}`)).toBe(false);
	});

	it("rejects non-media data: payloads", () => {
		expect(isClassifiableUrl(`data:text/html;base64,${"A".repeat(2048)}`)).toBe(false);
	});

	it("keeps blob: URLs", () => {
		expect(isClassifiableUrl("blob:https://example.com/1234")).toBe(true);
	});
});

describe("isTrackingPixel", () => {
	it("skips media under ~24 px on both axes", () => {
		expect(isTrackingPixel(1, 1)).toBe(true);
		expect(isTrackingPixel(10, 20)).toBe(true);
	});

	it("keeps media at or above 24 px on either axis", () => {
		expect(isTrackingPixel(24, 1)).toBe(false);
		expect(isTrackingPixel(1, 100)).toBe(false);
		expect(isTrackingPixel(300, 200)).toBe(false);
	});

	it("keeps unloaded media (0x0 — size unknown yet)", () => {
		expect(isTrackingPixel(0, 0)).toBe(false);
	});
});

describe("extractBackgroundImageUrl", () => {
	it("extracts quoted and unquoted url() values", () => {
		expect(extractBackgroundImageUrl('url("https://example.com/bg.png")')).toBe(
			"https://example.com/bg.png",
		);
		expect(extractBackgroundImageUrl("url('https://example.com/bg.png')")).toBe(
			"https://example.com/bg.png",
		);
		expect(extractBackgroundImageUrl("url(https://example.com/bg.png)")).toBe(
			"https://example.com/bg.png",
		);
	});

	it("resolves relative URLs against the document", () => {
		expect(extractBackgroundImageUrl("url(/img/bg.png)")).toMatch(/\/img\/bg\.png$/);
	});

	it("returns null when there is no url()", () => {
		expect(extractBackgroundImageUrl("none")).toBeNull();
		expect(extractBackgroundImageUrl("linear-gradient(red, blue)")).toBeNull();
	});
});

describe("imageUrl / videoUrl", () => {
	it("prefers currentSrc, falls back to src and data-src", () => {
		const img = document.createElement("img");
		img.src = "https://example.com/a.jpg";
		expect(imageUrl(img)).toBe("https://example.com/a.jpg");
		img.removeAttribute("src");
		img.setAttribute("data-src", "https://example.com/lazy.jpg");
		expect(imageUrl(img)).toBe("https://example.com/lazy.jpg");
	});

	it("prefers video poster over src", () => {
		const video = document.createElement("video");
		video.poster = "https://example.com/poster.jpg";
		video.src = "https://example.com/clip.mp4";
		expect(videoUrl(video)).toBe("https://example.com/poster.jpg");
	});
});

describe("candidateFor", () => {
	it("classifies a normal img as an image candidate", () => {
		const img = document.createElement("img");
		img.src = "https://example.com/a.jpg";
		const c = candidateFor(img, true);
		expect(c?.kind).toBe("image");
		expect(c?.url).toBe("https://example.com/a.jpg");
		expect(c?.visible).toBe(true);
	});

	it("rejects a tracking-pixel-sized img", () => {
		const img = document.createElement("img");
		img.src = "https://example.com/pixel.gif";
		img.width = 1;
		img.height = 1;
		expect(candidateFor(img, false)).toBeNull();
	});

	it("rejects an img whose only URL is an svg", () => {
		const img = document.createElement("img");
		img.src = "https://example.com/logo.svg";
		expect(candidateFor(img, false)).toBeNull();
	});

	it("keeps an img with no URL yet (lazy loading)", () => {
		const img = document.createElement("img");
		const c = candidateFor(img, false);
		expect(c?.kind).toBe("image");
		expect(c?.url).toBeNull();
	});

	it("extracts a background image URL from inline style", () => {
		// getComputedStyle resolves only for attached elements (both jsdom and
		// Chromium), which is also all discovery ever scans.
		const div = document.createElement("div");
		div.style.backgroundImage = "url('https://example.com/hero.jpg')";
		document.body.appendChild(div);
		const c = candidateFor(div, false);
		expect(c?.kind).toBe("background");
		expect(c?.url).toBe("https://example.com/hero.jpg");
		div.remove();
	});

	it("ignores elements without a background image", () => {
		const div = document.createElement("div");
		expect(candidateFor(div, false)).toBeNull();
	});

	it("classifies a video with its poster", () => {
		const video = document.createElement("video");
		video.poster = "https://example.com/p.jpg";
		const c = candidateFor(video, false);
		expect(c?.kind).toBe("video");
		expect(c?.url).toBe("https://example.com/p.jpg");
	});
});

describe("mediaElements", () => {
	it("yields img, video and possible background carriers once each", () => {
		document.body.innerHTML = `
			<div id="a"><img src="https://example.com/x.jpg"></div>
			<video id="v" poster="https://example.com/p.jpg"></video>
			<p>text</p>`;
		const els = [...mediaElements(document.body)];
		const ids = els.map((el) => el.id || el.tagName);
		expect(ids).toContain("a"); // div is a bg carrier candidate
		expect(ids).toContain("v");
		expect(els.filter((el) => el.tagName === "IMG")).toHaveLength(1);
		expect(els.filter((el) => el.tagName === "P")).toHaveLength(0);
	});

	it("includes a root element that is itself an img", () => {
		const img = document.createElement("img");
		img.src = "https://example.com/a.jpg";
		expect([...mediaElements(img)]).toContain(img);
	});
});

describe("shadow DOM piercing", () => {
	it("shadowHosts yields open shadow hosts, recursing into nested roots", async () => {
		const { shadowHosts } = await import("../../src/content/discovery");
		document.body.innerHTML = `<div id="outer"></div>`;
		const outer = document.getElementById("outer") as HTMLElement;
		const sr1 = outer.attachShadow({ mode: "open" });
		sr1.innerHTML = `<section id="inner-host"></section>`;
		const inner = sr1.getElementById("inner-host") as HTMLElement;
		inner.attachShadow({ mode: "open" });
		const hosts = [...shadowHosts(document.body)];
		expect(hosts).toContain(outer);
		expect(hosts).toContain(inner);
	});

	it("deepQueryAll finds matches inside shadow roots", async () => {
		const { deepQueryAll } = await import("../../src/content/discovery");
		document.body.innerHTML = `<div id="host"></div><img id="light" src="https://x.test/l.jpg">`;
		const host = document.getElementById("host") as HTMLElement;
		const sr = host.attachShadow({ mode: "open" });
		sr.innerHTML = `<img id="deep" src="https://x.test/d.jpg">`;
		const found = [...deepQueryAll<HTMLElement>(document, "img")].map((el) => el.id);
		expect(found).toContain("light");
		expect(found).toContain("deep");
	});

	it("startDiscovery emits candidates for media inside shadow roots", async () => {
		const { startDiscovery } = await import("../../src/content/discovery");
		vi.stubGlobal(
			"IntersectionObserver",
			class {
				observe() {}
				unobserve() {}
				disconnect() {}
			},
		);
		document.body.innerHTML = `<div id="host"></div>`;
		const host = document.getElementById("host") as HTMLElement;
		const sr = host.attachShadow({ mode: "open" });
		sr.innerHTML = `<img src="https://x.test/shadow.jpg">`;
		const seen: string[] = [];
		const handle = startDiscovery(document, {
			onCandidate: (c) => seen.push(`${c.kind}:${c.url}`),
			onVisible: () => {},
		});
		expect(seen).toContain("image:https://x.test/shadow.jpg");
		// Mutations inside the shadow root are observed too.
		const img2 = document.createElement("img");
		img2.src = "https://x.test/later.jpg";
		sr.appendChild(img2);
		await vi.waitFor(() => {
			expect(seen).toContain("image:https://x.test/later.jpg");
		});
		handle.stop();
		vi.unstubAllGlobals();
	});
});
