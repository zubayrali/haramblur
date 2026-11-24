import { describe, expect, it } from "vitest";
import { isRtl } from "../../src/shared/i18n";

describe("isRtl", () => {
	it("flags RTL languages", () => {
		expect(isRtl("ar")).toBe(true);
		expect(isRtl("ur")).toBe(true);
		expect(isRtl("ar-SA")).toBe(true);
		expect(isRtl("he")).toBe(true);
	});
	it("flags LTR languages", () => {
		expect(isRtl("en")).toBe(false);
		expect(isRtl("tr")).toBe(false);
		expect(isRtl("en-US")).toBe(false);
	});
});
