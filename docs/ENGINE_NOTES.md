# Engine notes (from the research spike)

Verified in Chromium with onnxruntime-web 1.31.1 on WASM (8-core CPU, no GPU): model load 0.8 s, about 85 ms per image.

-   Model: `320n.onnx`, input `[1, 3, 320, 320]` float32 RGB in 0..1, letterboxed to the top-left with fill `rgb(114,114,114)`, scale `k = min(320 / w, 320 / h)`.
-   Output: `[1, 4 + 18, N]` (YOLOv8 layout, channel-major): rows 0-3 are `cx, cy, w, h` in input pixels, rows 4.. are class scores. Divide the box by `k` to map back to the source image. Apply per-class NMS (IoU 0.45).
-   Class order:
    `FEMALE_GENITALIA_COVERED, FACE_FEMALE, BUTTOCKS_EXPOSED, FEMALE_BREAST_EXPOSED, FEMALE_GENITALIA_EXPOSED, MALE_BREAST_EXPOSED, ANUS_EXPOSED, FEET_EXPOSED, BELLY_COVERED, FEET_COVERED, ARMPITS_COVERED, ARMPITS_EXPOSED, FACE_MALE, BELLY_EXPOSED, MALE_GENITALIA_EXPOSED, ANUS_COVERED, FEMALE_BREAST_COVERED, BUTTOCKS_COVERED`
-   `ort.env.wasm.wasmPaths` must be an absolute URL (e.g. `browser.runtime.getURL("/ort/")`). A relative path resolves against the ort module's own directory and 404s.
-   Spike results: female faces found in all 4 test photos of women (scores 0.67-0.85), including a 3D-rendered woman with a small face that the old extension missed; nothing above 0.3 on a plain gradient.
-   MediaPipe BlazeFace short-range (`blaze_face_short_range.tflite`, 229,746 bytes, SHA-256 `b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f`): 13-74 ms, but it missed the small faces in a group photo. Use only as an optional helper.

## Face pipeline (NudeNet + BlazeFace + FastFace)

NudeNet's `FACE_*` labels conflate "is there a face" with "whose face" and are noisy (male portraits scoring `FACE_FEMALE@0.8+`). The engine therefore runs a dedicated two-stage face pipeline on the whole-image pass:

-   `blaze-face.onnx` (535,842 bytes, SHA-256 `564740c5…f8a5`): BlazeFace short-range ONNX export with decode+NMS+thresholding fused in-graph (Apache-2.0). Input `image` `[1,3,128,128]` float32 RGB 0..1 (stretched, not letterboxed) plus scalar `conf_threshold`/`iou_threshold`/`max_detections`; output `selectedBoxes` `[1,N,16]` rows of `top_y, top_x, bot_y, bot_x` + 6 landmark pairs, no scores. Runs WASM at three confidence passes (0.4/0.55/0.7 → scores 0.45/0.65/0.85) to recover per-face confidence for the strictness slider.
-   `fastface-small-112.onnx` (5.1 MB, Apache-2.0): MobileNetV3 gender+age on a 25%-padded square crop; `gender_logits[1,2]` index 0 = female.
-   Ensemble-AND gate (`FACE_CORROBORATE_MIN = 0.3`): a detector box only becomes a synthetic `FACE_*` detection when a NudeNet `FACE_*` ≥0.3 overlaps it. BlazeFace alone false-positives confidently on animal faces (dog fixture ≥0.8); NudeNet alone has the gender problem. Synthetic detections replace overlapping NudeNet `FACE_*` labels; the rest pass through to FastFace refinement as before.
