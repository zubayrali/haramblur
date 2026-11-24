import { describe, expect, it } from "vitest";
import { LruCache } from "../../src/engine/cache";

describe("LruCache", () => {
	it("stores and returns values", () => {
		const c = new LruCache<string, number>(3);
		c.set("a", 1);
		expect(c.get("a")).toBe(1);
		expect(c.get("missing")).toBeUndefined();
	});

	it("evicts the oldest entry at capacity", () => {
		const c = new LruCache<string, number>(2);
		c.set("a", 1);
		c.set("b", 2);
		c.set("c", 3);
		expect(c.get("a")).toBeUndefined();
		expect(c.get("b")).toBe(2);
		expect(c.get("c")).toBe(3);
	});

	it("a get refreshes recency so the touched key survives eviction", () => {
		const c = new LruCache<string, number>(2);
		c.set("a", 1);
		c.set("b", 2);
		expect(c.get("a")).toBe(1);
		c.set("c", 3);
		expect(c.get("b")).toBeUndefined();
		expect(c.get("a")).toBe(1);
		expect(c.get("c")).toBe(3);
	});

	it("set on an existing key replaces without counting twice", () => {
		const c = new LruCache<string, number>(2);
		c.set("a", 1);
		c.set("a", 42);
		c.set("b", 2);
		expect(c.size).toBe(2);
		expect(c.get("a")).toBe(42);
	});

	it("respects a 500-entry cap", () => {
		const c = new LruCache<string, number>(500);
		for (let i = 0; i < 600; i++) c.set(`u${i}`, i);
		expect(c.size).toBe(500);
		expect(c.get("u0")).toBeUndefined();
		expect(c.get("u599")).toBe(599);
	});
});
