/**
 * nsfwjs MobileNetV2 whole-image classifier glue (MIT weights, uint8 ONNX).
 *
 * The original extension relied on nsfwjs's 5 whole-image classes
 * (Drawing/Hentai/Neutral/Porn/Sexy) for suggestive/anime recall that
 * NudeNet's explicit-body-part classes miss. This module turns the model's
 * logits into a Detection-friendly margin score:
 *
 *   nsfw = max(Porn, Hentai, Sexy)   sfw = max(Drawing, Neutral)
 *   margin = nsfw - sfw              (range roughly -1..1)
 *
 * A positive margin means the image leans NSFW; `policy.nsfwThreshold`
 * decides whether it is enough to blur. We emit at most one detection —
 * `NSFW_EXPLICIT` when Porn/Hentai leads, else `NSFW_SUGGESTIVE` — with the
 * margin as its score and a full-frame box (there is nothing to localize).
 */
import type { NsfwLabel } from "../shared/labels";
import type { Detection } from "../shared/protocol";

/** nsfwjs MobileNetV2 class order (verified against upstream model defs). */
export const NSFW_CLASSES = ["Drawing", "Hentai", "Neutral", "Porn", "Sexy"] as const;
export const NSFW_INPUT_SIZE = 224;

const CLASS = {
	drawing: 0,
	hentai: 1,
	neutral: 2,
	porn: 3,
	sexy: 4,
} as const;

/** Softmax over logits. */
export function softmax(logits: ArrayLike<number>): number[] {
	const m = Math.max(...Array.from(logits));
	const exps = Array.from(logits, (v) => Math.exp(v - m));
	const sum = exps.reduce((a, b) => a + b, 0);
	return exps.map((e) => e / sum);
}

export interface NsfwVerdict {
	label: NsfwLabel;
	/** nsfw − sfw probability margin; only meaningful when > 0. */
	margin: number;
	probs: number[];
}

/**
 * Reduce class probabilities to a single NSFW verdict, or null when the
 * image does not lean NSFW at all (`margin <= 0`) or the margin is trivially
 * small (`< minMargin`, default 0.02 — below that it's noise).
 */
export function nsfwVerdict(
	probs: readonly number[],
	opts: { minMargin?: number } = {},
): NsfwVerdict | null {
	const minMargin = opts.minMargin ?? 0.02;
	const p = (i: number) => probs[i] ?? 0;
	const explicit = Math.max(p(CLASS.porn), p(CLASS.hentai));
	const suggestive = p(CLASS.sexy);
	const nsfw = Math.max(explicit, suggestive);
	const sfw = Math.max(p(CLASS.drawing), p(CLASS.neutral));
	const margin = nsfw - sfw;
	if (margin <= minMargin) return null;
	return {
		label: explicit >= suggestive ? "NSFW_EXPLICIT" : "NSFW_SUGGESTIVE",
		margin,
		probs: [...probs],
	};
}

/** Wrap a verdict as a full-frame Detection for the policy pipeline. */
export function nsfwDetection(v: NsfwVerdict): Detection {
	return { label: v.label, score: v.margin, box: [0, 0, 1, 1] };
}
