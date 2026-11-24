/**
 * XState 5 machine for the engine lifecycle (docs/PLAN.md "State machines").
 *
 * `booting` → `ready` (WebGPU) | `degraded` (WASM fallback) | `failed`.
 * A worker that dies or fails to init restarts once (`MAX_RESTARTS`), then the
 * machine stays `failed`. The host drives it with READY/CRASH events and reads
 * `engineStatus` from the snapshot.
 */
import { assign, setup } from "xstate";
import type { EngineBackend } from "../shared/protocol";

/** How many times a dead/failed worker is restarted before `failed` sticks. */
export const MAX_RESTARTS = 1;

export interface EngineContext {
	restarts: number;
	backend: EngineBackend | null;
	error: string | null;
}

export type EngineEvent =
	| { type: "READY"; backend: EngineBackend }
	| { type: "CRASH"; error: string };

export const engineMachine = setup({
	types: { context: {} as EngineContext, events: {} as EngineEvent },
}).createMachine({
	id: "hbEngine",
	initial: "booting",
	context: { restarts: 0, backend: null, error: null },
	states: {
		booting: {
			on: {
				READY: [
					{
						guard: ({ event }) => event.backend === "webgpu",
						target: "ready",
						actions: assign({ backend: ({ event }) => event.backend }),
					},
					{
						target: "degraded",
						actions: assign({ backend: ({ event }) => event.backend }),
					},
				],
				CRASH: [
					{
						guard: ({ context }) => context.restarts < MAX_RESTARTS,
						target: "booting",
						actions: assign({
							restarts: ({ context }) => context.restarts + 1,
							backend: () => null,
							error: ({ event }) => event.error,
						}),
					},
					{
						target: "failed",
						actions: assign({ error: ({ event }) => event.error }),
					},
				],
			},
		},
		ready: {
			on: {
				CRASH: [
					{
						guard: ({ context }) => context.restarts < MAX_RESTARTS,
						target: "booting",
						actions: assign({
							restarts: ({ context }) => context.restarts + 1,
							backend: () => null,
							error: ({ event }) => event.error,
						}),
					},
					{
						target: "failed",
						actions: assign({ error: ({ event }) => event.error }),
					},
				],
			},
		},
		degraded: {
			on: {
				CRASH: [
					{
						guard: ({ context }) => context.restarts < MAX_RESTARTS,
						target: "booting",
						actions: assign({
							restarts: ({ context }) => context.restarts + 1,
							backend: () => null,
							error: ({ event }) => event.error,
						}),
					},
					{
						target: "failed",
						actions: assign({ error: ({ event }) => event.error }),
					},
				],
			},
		},
		failed: {
			type: "final",
		},
	},
});
