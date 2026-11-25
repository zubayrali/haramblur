/**
 * Typed ONNX model slots (docs/PLAN.md Phase 4). Each slot has a fixed
 * tensor contract — the engine's pre/post-processing is written against it,
 * so an override must be an export of the same architecture, not an
 * arbitrary graph.
 */
import type { I18nKey } from "./i18n";

export type ModelSlot = "nudenet" | "fastface" | "yunet" | "nsfw" | "person";

export interface ModelSlotInfo {
	id: ModelSlot;
	/** Filename of the bundled model shipped in `public/models/`. */
	bundledFile: string;
	/** i18n key for the slot's display name. */
	labelKey: I18nKey;
	/** i18n key describing the expected tensor contract. */
	contractKey: I18nKey;
}

export const MODEL_SLOTS: readonly ModelSlotInfo[] = [
	{
		id: "nudenet",
		bundledFile: "320n.onnx",
		labelKey: "modelNudenet",
		contractKey: "modelNudenetContract",
	},
	{
		id: "fastface",
		bundledFile: "fastface-small-112.onnx",
		labelKey: "modelFastface",
		contractKey: "modelFastfaceContract",
	},
	{
		id: "yunet",
		bundledFile: "yunet-2023mar.onnx",
		labelKey: "modelYunet",
		contractKey: "modelYunetContract",
	},
	{
		id: "nsfw",
		bundledFile: "nsfwjs-u8.onnx",
		labelKey: "modelNsfw",
		contractKey: "modelNsfwContract",
	},
	{
		id: "person",
		bundledFile: "yolov8n-seg-fp16.onnx",
		labelKey: "modelPerson",
		contractKey: "modelPersonContract",
	},
];

export function slotInfo(slot: ModelSlot): ModelSlotInfo {
	const info = MODEL_SLOTS.find((s) => s.id === slot);
	if (info == null) throw new Error(`unknown model slot: ${slot}`);
	return info;
}
