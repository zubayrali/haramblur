import { describe, expect, it } from "vitest";
import {
	isAllowlisted,
	normalizeHostEntry,
	reasonsFromJson,
	reasonsToJson,
	redecide,
} from "../../src/content/site";
import type { Detection } from "../../src/shared/protocol";
import { DEFAULT_SETTINGS } from "../../src/shared/settings";

describe("isAllowlisted", () => {
	it("matches a hostname exactly", () => {
		expect(isAllowlisted("example.com", ["example.com"])).toBe(true);
	});

	it("covers subdomains of an entry", () => {
		expect(isAllowlisted("www.example.com", ["example.com"])).toBe(true);
		expect(isAllowlisted("a.b.example.com", ["example.com"])).toBe(true);
	});

	it("does not match partial suffixes", () => {
		expect(isAllowlisted("notexample.com", ["example.com"])).toBe(false);
		expect(isAllowlisted("example.com.evil.org", ["example.com"])).toBe(false);
	});

	it("accepts entries with scheme, port or path", () => {
		expect(isAllowlisted("example.com", ["https://example.com:8443/x"])).toBe(true);
	});

	it("is case-insensitive and ignores empty entries", () => {
		expect(isAllowlisted("EXAMPLE.com", ["Example.COM"])).toBe(true);
		expect(isAllowlisted("a.com", ["", "  "])).toBe(false);
	});
});

describe("normalizeHostEntry", () => {
	it("strips scheme, port and path, lowercases", () => {
		expect(normalizeHostEntry("https://WWW.Example.COM:443/path?q=1")).toBe("www.example.com");
	});
});

const det = (label: Detection["label"], score: number): Detection => ({
	label,
	score,
	box: [0.1, 0.2, 0.3, 0.4],
});

describe("data-hb-reasons round-trip", () => {
	it("serializes detections to [{label, score}] JSON and back", () => {
		const json = reasonsToJson([det("FACE_FEMALE", 0.8), det("BELLY_COVERED", 0.4)]);
		expect(JSON.parse(json)).toEqual([
			{ label: "FACE_FEMALE", score: 0.8 },
			{ label: "BELLY_COVERED", score: 0.4 },
		]);
		const back = reasonsFromJson(json);
		expect(back).toHaveLength(2);
		expect(back[0]?.label).toBe("FACE_FEMALE");
		expect(back[0]?.score).toBe(0.8);
	});

	it("parses corrupt JSON to an empty list", () => {
		expect(reasonsFromJson("not json")).toEqual([]);
		expect(reasonsFromJson('[{"nope":1}]')).toEqual([]);
	});
});

describe("redecide (policy application on stored detections)", () => {
	it("a clear verdict flips to blur when strictness rises", () => {
		const json = reasonsToJson([det("FACE_FEMALE", 0.3)]);
		expect(redecide(json, { ...DEFAULT_SETTINGS, strictness: 0 })).toBe("clear");
		expect(redecide(json, { ...DEFAULT_SETTINGS, strictness: 1 })).toBe("blur");
	});

	it("a blur verdict flips to clear when the gender toggle is disabled", () => {
		const json = reasonsToJson([det("FACE_FEMALE", 0.9)]);
		expect(redecide(json, DEFAULT_SETTINGS)).toBe("blur");
		expect(redecide(json, { ...DEFAULT_SETTINGS, blurFemale: false })).toBe("clear");
	});

	it("everything clears when the extension is off", () => {
		const json = reasonsToJson([det("ANUS_EXPOSED", 0.99)]);
		expect(redecide(json, { ...DEFAULT_SETTINGS, status: false })).toBe("clear");
	});
});
