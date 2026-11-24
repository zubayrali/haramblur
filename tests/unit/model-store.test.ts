import { describe, expect, it } from "vitest";
import { ModelStore, memoryBackend, sha256Hex } from "../../src/engine/model-store";

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

describe("sha256Hex", () => {
	it("matches the known SHA-256 of empty input", async () => {
		expect(await sha256Hex(new Uint8Array(0))).toBe(
			"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		);
	});
});
