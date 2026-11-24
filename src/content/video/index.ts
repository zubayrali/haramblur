import { sendMessage } from "../../shared/protocol";
import { getSettings, watchSettings } from "../../shared/settings-store";
import { VideoManager } from "./manager";
import type { VideoManagerDeps } from "./types";

export type { CapturedFrame } from "./frames";
export {
	captureFrame,
	FRAME_MIME,
	HASH_GRID,
	hashDistance,
	MAX_FRAME_EDGE,
	scaledSize,
} from "./frames";
export { MAX_CONCURRENT_VIDEOS, VideoManager } from "./manager";
export type {
	SamplerContext,
	SamplerEvent,
	SamplerState,
} from "./sampler";
export {
	applyVerdict,
	BASE_FPS,
	BOOST_FPS,
	BOOST_MS,
	HIGH_CONFIDENCE_SCORE,
	HYSTERESIS_WINDOW,
	isHighConfidence,
	SCENE_CHANGE_DISTANCE,
	STABLE_AFTER,
	STABLE_FPS,
	samplerMachine,
	VideoSampler,
} from "./sampler";
export type {
	ClassifyFrame,
	ClassifyFrameArgs,
	SamplerEnvironment,
	VideoManagerDeps,
	VideoSamplerDeps,
} from "./types";

/**
 * Wire the video pipeline to the real extension transport and settings.
 * `src/content/index.ts` (WP2) calls `createVideoManager().start()` and feeds
 * it `<video>` elements from discovery.
 */
export function createVideoManager(overrides: Partial<VideoManagerDeps> = {}): VideoManager {
	return new VideoManager({
		classifyFrame: (args) => sendMessage("classifyFrame", args),
		getSettings,
		watchSettings,
		...overrides,
	});
}
