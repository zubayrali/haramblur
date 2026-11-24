/**
 * OPFS-backed override store for downloaded/imported ONNX models
 * (docs/PLAN.md Phase 4). The bundled models in `public/models/` stay the
 * default; a slot override in OPFS wins at engine init. All weights stay
 * local — user-initiated file picks only, no runtime CDN fetches.
 */
import type { ModelSlot } from "../shared/model-slots";
import { MODEL_SLOTS } from "../shared/model-slots";

export interface ModelMeta {
	sha256: string;
	size: number;
	installedAt: number;
	fileName: string;
}

export interface InstalledModel {
	slot: ModelSlot;
	meta: ModelMeta;
}

/** Minimal key→bytes backend so the store is testable without OPFS. */
export interface ModelBackend {
	get(name: string): Promise<Uint8Array | null>;
	put(name: string, data: Uint8Array): Promise<void>;
	remove(name: string): Promise<void>;
	list(): Promise<string[]>;
}

const MODEL_EXT = ".onnx";
const META_EXT = ".json";

export class ModelStore {
	constructor(private readonly backend: ModelBackend) {}

	private modelName(slot: ModelSlot): string {
		return `${slot}${MODEL_EXT}`;
	}
	private metaName(slot: ModelSlot): string {
		return `${slot}${META_EXT}`;
	}

	async list(): Promise<InstalledModel[]> {
		const names = await this.backend.list();
		const out: InstalledModel[] = [];
		for (const slot of MODEL_SLOTS) {
			if (!names.includes(this.modelName(slot.id))) continue;
			const raw = await this.backend.get(this.metaName(slot.id));
			let meta: ModelMeta | null = null;
			if (raw != null) {
				try {
					meta = JSON.parse(new TextDecoder().decode(raw)) as ModelMeta;
				} catch {
					meta = null;
				}
			}
			if (meta == null) {
				const bytes = await this.backend.get(this.modelName(slot.id));
				meta = {
					sha256: "",
					size: bytes?.length ?? 0,
					installedAt: 0,
					fileName: "",
				};
			}
			out.push({ slot: slot.id, meta });
		}
		return out;
	}

	async getBytes(slot: ModelSlot): Promise<Uint8Array | null> {
		return this.backend.get(this.modelName(slot));
	}

	async put(slot: ModelSlot, bytes: Uint8Array, fileName: string): Promise<ModelMeta> {
		const meta: ModelMeta = {
			sha256: await sha256Hex(bytes),
			size: bytes.length,
			installedAt: Date.now(),
			fileName,
		};
		await this.backend.put(this.modelName(slot), bytes);
		await this.backend.put(this.metaName(slot), new TextEncoder().encode(JSON.stringify(meta)));
		return meta;
	}

	async remove(slot: ModelSlot): Promise<void> {
		await this.backend.remove(this.modelName(slot));
		await this.backend.remove(this.metaName(slot));
	}
}

export async function sha256Hex(data: Uint8Array): Promise<string> {
	const digest = await crypto.subtle.digest("SHA-256", data as BufferSource);
	return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** In-memory backend — unit tests and non-OPFS fallbacks. */
export function memoryBackend(): ModelBackend & { data: Map<string, Uint8Array> } {
	const data = new Map<string, Uint8Array>();
	return {
		data,
		async get(name) {
			return data.get(name) ?? null;
		},
		async put(name, bytes) {
			data.set(name, bytes);
		},
		async remove(name) {
			data.delete(name);
		},
		async list() {
			return [...data.keys()];
		},
	};
}

interface FileHandleLike {
	getFile(): Promise<{ arrayBuffer(): Promise<ArrayBuffer> }>;
	createWritable(): Promise<{
		write(data: Uint8Array): Promise<void>;
		close(): Promise<void>;
	}>;
}

interface DirHandleLike {
	getFileHandle(name: string, opts?: { create?: boolean }): Promise<FileHandleLike>;
	getDirectoryHandle(name: string, opts?: { create?: boolean }): Promise<DirHandleLike>;
	removeEntry(name: string): Promise<void>;
}

interface StorageManagerLike {
	getDirectory(): Promise<DirHandleLike>;
}

/** OPFS backend — `navigator.storage.getDirectory()` under `hb-models/`. */
export function opfsBackend(dirName = "hb-models"): ModelBackend {
	let dirPromise: Promise<DirHandleLike> | null = null;
	const dir = () => {
		dirPromise ??= (async () => {
			const root = await (navigator.storage as StorageManagerLike).getDirectory();
			return root.getDirectoryHandle(dirName, { create: true });
		})();
		return dirPromise;
	};
	return {
		async get(name) {
			try {
				const d = await dir();
				const fh = await d.getFileHandle(name);
				const file = await fh.getFile();
				return new Uint8Array(await file.arrayBuffer());
			} catch {
				return null;
			}
		},
		async put(name, data) {
			const d = await dir();
			const fh = await d.getFileHandle(name, { create: true });
			const w = await fh.createWritable();
			await w.write(data);
			await w.close();
		},
		async remove(name) {
			try {
				const d = await dir();
				await d.removeEntry(name);
			} catch {
				// Already gone.
			}
		},
		async list() {
			// OPFS directory listing needs async iteration; enumerate known
			// slot names instead — the store only ever writes slot files.
			const names: string[] = [];
			for await (const handle of (await dir()) as unknown as AsyncIterable<{
				kind: string;
				name: string;
			}>) {
				if (handle.kind === "file") names.push(handle.name);
			}
			return names;
		},
	};
}

let shared: ModelStore | null = null;

/** Shared OPFS-backed store for pages/workers that support it. */
export function getModelStore(): ModelStore {
	shared ??= new ModelStore(opfsBackend());
	return shared;
}

/** Slot → bytes for every installed override (engine init path). */
export async function loadOverrides(
	store: ModelStore = getModelStore(),
): Promise<Partial<Record<ModelSlot, Uint8Array>>> {
	const out: Partial<Record<ModelSlot, Uint8Array>> = {};
	for (const slot of MODEL_SLOTS) {
		const bytes = await store.getBytes(slot.id);
		if (bytes != null) out[slot.id] = bytes;
	}
	return out;
}
