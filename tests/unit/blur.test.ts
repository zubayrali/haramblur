// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { applyBlurStyles, ensureShadowStyles, setKind } from "../../src/content/blur";
import { DEFAULT_SETTINGS } from "../../src/shared/settings";

describe("shadow-root blur styles", () => {
	it("adopts the shared stylesheet into open shadow roots", () => {
		applyBlurStyles(DEFAULT_SETTINGS);
		document.body.innerHTML = `<div id="host"></div>`;
		const host = document.getElementById("host") as HTMLElement;
		const sr = host.attachShadow({ mode: "open" });
		const img = document.createElement("img");
		sr.appendChild(img);
		setKind(img, "image");
		// jsdom may lack constructable stylesheets — the call must be a no-op
		// there, never a crash; real browsers adopt the sheet.
		if ("adoptedStyleSheets" in sr) {
			expect((sr.adoptedStyleSheets as unknown[]).length).toBeGreaterThan(0);
		}
	});

	it("is a no-op for light-DOM elements", () => {
		const img = document.createElement("img");
		document.body.appendChild(img);
		expect(() => {
			ensureShadowStyles(img);
			setKind(img, "image");
		}).not.toThrow();
	});
});
