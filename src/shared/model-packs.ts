import type { I18nKey } from "./i18n";
import type { ModelSlot } from "./model-slots";

/**
 * Curated downloadable model packs (docs/PLAN.md Phase 4). The catalog IS
 * the suitability guardrail: every entry names its slot, so its tensor
 * contract is fixed — users can never import an incompatible graph through
 * this path (arbitrary local files remain possible via the slot's Import
 * button, at their own risk).
 */
export interface ModelPack {
	id: string;
	slot: ModelSlot;
	labelKey: I18nKey;
	descKey: I18nKey;
	/** HTTPS URL the bytes are fetched from (user-initiated only). */
	url: string;
	/** Pinned SHA-256 — a mismatch aborts the install. */
	sha256: string;
	sizeBytes: number;
}

export const MODEL_PACKS: readonly ModelPack[] = [
	{
		id: "yunet-2023mar-int8",
		slot: "yunet",
		labelKey: "packYunetInt8",
		descKey: "packYunetInt8Desc",
		url: "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar_int8.onnx",
		sha256: "321aa5a6afabf7ecc46a3d06bfab2b579dc96eb5c3be7edd365fa04502ad9294",
		sizeBytes: 100_416,
	},
	{
		id: "fastface-small-112-int8",
		slot: "fastface",
		labelKey: "packFastfaceInt8",
		descKey: "packFastfaceInt8Desc",
		url: "https://huggingface.co/iFurySt/fastface/resolve/main/models/fastface-small-112/model_int8_static.onnx",
		sha256: "6bd6201ce83bfc44a81ce77f43f4db5fff85d14c00c050844d9d2615c007ccee",
		sizeBytes: 1_616_311,
	},
];

export function packsForSlot(slot: ModelSlot): ModelPack[] {
	return MODEL_PACKS.filter((p) => p.slot === slot);
}
