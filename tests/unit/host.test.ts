import { describe, expect, it } from "vitest";
import { hostnameFromUrl, normalizeHostname, toggleAllowlistHost } from "../../src/ui/host";

describe("hostnameFromUrl", () => {
	it("extracts the hostname from http(s) URLs", () => {
		expect(hostnameFromUrl("https://www.example.com/path?q=1")).toBe("www.example.com");
		expect(hostnameFromUrl("http://sub.example.co.uk")).toBe("sub.example.co.uk");
	});
	it("returns null for non-web pages", () => {
		expect(hostnameFromUrl("chrome://extensions")).toBeNull();
		expect(hostnameFromUrl("chrome-extension://abc/popup.html")).toBeNull();
		expect(hostnameFromUrl("about:blank")).toBeNull();
		expect(hostnameFromUrl(undefined)).toBeNull();
		expect(hostnameFromUrl("not a url")).toBeNull();
	});
});

describe("normalizeHostname", () => {
	it("accepts bare hostnames and full URLs", () => {
		expect(normalizeHostname("example.com")).toBe("example.com");
		expect(normalizeHostname("  Example.COM  ")).toBe("example.com");
		expect(normalizeHostname("https://example.com/some/page")).toBe("example.com");
		expect(normalizeHostname("example.com:8080/x")).toBe("example.com");
		expect(normalizeHostname("example.com.")).toBe("example.com");
	});
	it("rejects invalid input", () => {
		expect(normalizeHostname("")).toBeNull();
		expect(normalizeHostname("   ")).toBeNull();
		expect(normalizeHostname("not a host")).toBeNull();
		expect(normalizeHostname("-bad-.com")).toBeNull();
		expect(normalizeHostname("http://")).toBeNull();
	});
});

describe("toggleAllowlistHost", () => {
	it("adds missing hosts and removes present ones, sorted", () => {
		expect(toggleAllowlistHost(["b.com"], "a.com")).toEqual(["a.com", "b.com"]);
		expect(toggleAllowlistHost(["a.com", "b.com"], "a.com")).toEqual(["b.com"]);
	});
});
