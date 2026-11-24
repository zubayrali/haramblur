# Engine notes (from the research spike)

Verified in Chromium with onnxruntime-web 1.31.1 on WASM (8-core CPU, no GPU): model load 0.8 s, about 85 ms per image.

## Models (all bundled, fetched by `pnpm models:fetch`)

-   `320n.onnx` — NudeNet 320n (YOLOv8n, AGPL-3.0), the region detector. Input `[1, 3, 320, 320]` float32 RGB in 0..1, letterboxed to the top-left with fill `rgb(114,114,114)`, scale `k = min(320 / w, 320 / h)`. Output `[1, 4 + 18, N]` (YOLOv8 layout, channel-major): rows 0-3 are `cx, cy, w, h` in input pixels, rows 4.. are class scores. Divide the box by `k` to map back to the source image. Apply per-class NMS (IoU 0.45). Class order:
    `FEMALE_GENITALIA_COVERED, FACE_FEMALE, BUTTOCKS_EXPOSED, FEMALE_BREAST_EXPOSED, FEMALE_GENITALIA_EXPOSED, MALE_BREAST_EXPOSED, ANUS_EXPOSED, FEET_EXPOSED, BELLY_COVERED, FEET_COVERED, ARMPITS_COVERED, ARMPITS_EXPOSED, FACE_MALE, BELLY_EXPOSED, MALE_GENITALIA_EXPOSED, ANUS_COVERED, FEMALE_BREAST_COVERED, BUTTOCKS_COVERED`
-   `yunet-2023mar.onnx` — OpenCV YuNet face detector (Apache-2.0, opencv_zoo, 232,589 bytes). Replaces MediaPipe BlazeFace: one letterboxed 640×640 pass, BGR 0..255, strides 8/16/32, score `sqrt(cls·obj)`, 5 landmarks per face. Detections stand alone — no NudeNet corroboration gate.
-   `fastface-small-112.onnx` — FastFace MobileNetV3 gender+age (Apache-2.0). Runs on landmark-aligned square crops (eyes leveled, centered, scaled by inter-ocular distance). Below the relabel gate (0.7; 0.85 for YuNet synthetics) the gender claim is unverifiable → the face collapses to the neutral `FACE_PERSON` label, which suppresses NudeNet's noisy `FACE_*` on the same box without asserting a gender.
-   `nsfwjs-u8.onnx` — nsfwjs MobileNetV2 uint8 (MIT weights). Whole-image Drawing/Hentai/Neutral/Porn/Sexy at 224×224 RGB NHWC; emits `NSFW_EXPLICIT`/`NSFW_SUGGESTIVE` as a full-frame detection. Runs only on the whole-image pass; skipped when the pass already decided `blur` **unless region blur is on** — an NSFW verdict's full-frame box still escalates to whole-element blur.
-   `ort.env.wasm.wasmPaths` must be an absolute URL (e.g. `browser.runtime.getURL("/ort/")`). A relative path resolves against the ort module's own directory and 404s.
-   The `.jsep` WASM variant is kept alongside the plain one: WebGPU EP needs it, WASM EP needs the plain binary.

## Spike results

-   NudeNet: female faces found in all 4 test photos of women (scores 0.67-0.85), including a 3D-rendered woman with a small face that the old extension missed; nothing above 0.3 on a plain gradient.
-   MediaPipe BlazeFace short-range (`blaze_face_short_range.tflite`): 13-74 ms, but it missed the small faces in a group photo — dropped in favour of YuNet.
