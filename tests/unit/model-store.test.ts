import { describe, expect, it } from "vitest";
import { downloadPack, ModelStore, memoryBackend, sha256Hex } from "../../src/engine/model-store";
import { MODEL_PACKS, packsForSlot } from "../../src/shared/model-packs";
import { MODEL_SLOTS } from "../../src/shared/model-slots";

const bytes = (n: number, fill = 1) => new Uint8Array(n).fill(fill);

describe("ModelStore", () => {
	it("put/get/list/remove round-trips a slot", async () => {
		const store = new ModelStore(memoryBackend());
		const meta = await store.put("nudenet", bytes(100), "640m.onnx");
		expect(meta.size).toBe(100);
		expect(meta.sha256).toHaveLength(64);
		expect(meta.fileName).toBe("640m.onnx");

		const got = await store.getBytes("nudenet");
		expect(got).not.toBeNull();
		expect(got!.length).toBe(100);

		const list = await store.list();
		expect(list).toHaveLength(1);
		expect(list[0]!.slot).toBe("nudenet");
		expect(list[0]!.meta.sha256).toBe(meta.sha256);

		await store.remove("nudenet");
		expect(await store.getBytes("nudenet")).toBeNull();
		expect(await store.list()).toHaveLength(0);
	});

	it("ignores orphan meta files and unknown slots", async () => {
		const backend = memoryBackend();
		await backend.put("rogue.onnx", bytes(5));
		const store = new ModelStore(backend);
		expect(await store.list()).toHaveLength(0);
	});

	it("falls back to size-only meta when the json is missing", async () => {
		const backend = memoryBackend();
		await backend.put("yunet.onnx", bytes(42));
		const store = new ModelStore(backend);
		const list = await store.list();
		expect(list[0]!.meta.size).toBe(42);
		expect(list[0]!.meta.sha256).toBe("");
	});
});

describe("MODEL_PACKS catalog", () => {
	it("every pack targets a real slot over https with a pinned hash", () => {
		const slots = new Set(MODEL_SLOTS.map((s) => s.id));
		for (const p of MODEL_PACKS) {
			expect(slots.has(p.slot)).toBe(true);
			expect(p.url.startsWith("https://")).toBe(true);
			expect(p.sha256).toMatch(/^[0-9a-f]{64}$/);
			expect(p.sizeBytes).toBeGreaterThan(0);
		}
	});

	it("packsForSlot only returns packs for that slot", () => {
		for (const s of MODEL_SLOTS) {
			for (const p of packsForSlot(s.id)) expect(p.slot).toBe(s.id);
		}
	});
});

describe("downloadPack", () => {
	const data = new TextEncoder().encode("fake-onnx-payload");
	const pack = {
		id: "test-pack",
		slot: "yunet" as const,
		labelKey: "modelDownload" as const,
		descKey: "modelDownloading" as const,
		url: "https://example.test/model.onnx",
		sizeBytes: data.length,
		sha256: "",
	};
	const okFetcher = async () =>
		new Response(data.slice().buffer as ArrayBuffer, {
			headers: { "content-length": String(data.length) },
		});

	it("verifies then stores a good download", async () => {
		const store = new ModelStore(memoryBackend());
		const p = { ...pack, sha256: await sha256Hex(data) };
		const meta = await downloadPack(p, store, undefined, okFetcher);
		expect(meta.sha256).toBe(p.sha256);
		expect([...(await store.getBytes("yunet"))!]).toEqual([...data]);
	});

	it("rejects a checksum mismatch", async () => {
		const store = new ModelStore(memoryBackend());
		await expect(
			downloadPack({ ...pack, sha256: "0".repeat(64) }, store, undefined, okFetcher),
		).rejects.toThrow("checksum");
		expect(await store.getBytes("yunet")).toBeNull();
	});

	it("rejects a size mismatch", async () => {
		const store = new ModelStore(memoryBackend());
		const bad = { ...pack, sizeBytes: data.length + 1, sha256: await sha256Hex(data) };
		await expect(downloadPack(bad, store, undefined, okFetcher)).rejects.toThrow("size");
	});

	it("rejects HTTP errors", async () => {
		const store = new ModelStore(memoryBackend());
		const fail = async () => new Response("nope", { status: 404 });
		await expect(
			downloadPack({ ...pack, sha256: await sha256Hex(data) }, store, undefined, fail),
		).rejects.toThrow("404");
	});
});

describe("sha256Hex", () => {
	it("matches the known SHA-256 of empty input", async () => {
		expect(await sha256Hex(new Uint8Array(0))).toBe(
			"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		);
	});
});
