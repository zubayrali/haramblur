import { describe, expect, it } from "vitest";
import { createActor } from "xstate";
import { engineMachine, MAX_RESTARTS } from "../../src/engine/lifecycle";

describe("engine lifecycle machine", () => {
	it("starts booting and reaches ready on WebGPU", () => {
		const actor = createActor(engineMachine).start();
		expect(actor.getSnapshot().value).toBe("booting");
		actor.send({ type: "READY", backend: "webgpu" });
		const snap = actor.getSnapshot();
		expect(snap.value).toBe("ready");
		expect(snap.context.backend).toBe("webgpu");
	});

	it("maps a WASM init to degraded", () => {
		const actor = createActor(engineMachine).start();
		actor.send({ type: "READY", backend: "wasm" });
		const snap = actor.getSnapshot();
		expect(snap.value).toBe("degraded");
		expect(snap.context.backend).toBe("wasm");
	});

	it("restarts once after a crash, then stays failed", () => {
		const actor = createActor(engineMachine).start();
		actor.send({ type: "READY", backend: "wasm" });
		actor.send({ type: "CRASH", error: "worker died" });
		let snap = actor.getSnapshot();
		expect(snap.value).toBe("booting");
		expect(snap.context.restarts).toBe(MAX_RESTARTS);
		// Second crash → failed for good.
		actor.send({ type: "CRASH", error: "worker died again" });
		snap = actor.getSnapshot();
		expect(snap.value).toBe("failed");
		expect(snap.context.error).toBe("worker died again");
	});

	it("a failed init also consumes the single restart", () => {
		const actor = createActor(engineMachine).start();
		actor.send({ type: "CRASH", error: "init failed" });
		expect(actor.getSnapshot().value).toBe("booting");
		actor.send({ type: "CRASH", error: "init failed again" });
		expect(actor.getSnapshot().value).toBe("failed");
	});

	it("keeps the last error visible on status", () => {
		const actor = createActor(engineMachine).start();
		actor.send({ type: "CRASH", error: "boom" });
		expect(actor.getSnapshot().context.error).toBe("boom");
	});
});
