# Competitive parity study — Porda-AI, upstream HaramBlur 0.6.11, and us

Status: research report. Supersedes the comparison table at
`docs/DETECTION_PARITY.md:14-22`, which is wrong in three material respects
(documented in §3.1). Builds on `docs/reviews/dp6-eval-opus-4-8.md` and
`docs/reviews/dp6-eval-opus-5.md` — their conclusions are not re-litigated here,
only validated or refuted against what the competitors' code actually does.

**Date of study:** 2026-11-25.

---

## 0. Scope, method, and what the evidence does not cover

### Artifacts read

| Target | Artifact | State |
| --- | --- | --- |
| Porda-AI | `/Users/fxwalken/Documents/GitHub/Porda-AI` — Python/PyQt5 desktop app, 20 files, 3 213 LOC + bundled Darknet weights | Full source |
| Upstream HaramBlur | `/tmp/hb-upstream-crx` — unpacked CRX v0.6.11, MV3 | Minified bundles + encrypted model blobs |
| Ours | `/Users/fxwalken/Documents/GitHub/haramblur` | Full source |

Minified bundles are single-line; citations are given as `file:line (byte N)`
so the offset can be recovered with `python3 -c "print(open(f).read()[N:N+600])"`.

### Evidence gaps — stated, not papered over

1. **Porda's browser extension source is not in this repo.** `README.md:40-44`
   points at a separate repo `hijam-git/porda-ai-extension` and a Chrome Web
   Store listing; neither is cloned locally. Everything in Part 1 describes the
   **desktop screen-overlay app**. The extension may or may not share the model.
2. **Porda's shipped binary is newer than its source.** `version.json` declares
   v2.2.3 (published 2026-10-20, 96 MB Windows installer) but the last commit
   touching `main.py`/`settings/` is `1fa42cb`, 2025-05-23. The v2.x detection
   changes advertised in the README banner ("faster detection") are **not in the
   open source**. Part 1 therefore describes the ~v1.x/2.0-era engine.
3. **Upstream's detector weights were not decrypted.** The obfuscation is
   trivially reversible (§2.2) but `src/mdls/MODELS_LICENSE.txt:12` forbids
   weight extraction, and we are AGPL — ingesting their weights would be both a
   licence violation and useless to us. Architecture in §2.1 is inferred from
   the *postprocessing code*, which is unambiguous, plus declared input shapes.
4. **Upstream's class semantics are partially inferred.** Class names are
   obfuscated (`x7kQ`, `m3Jz`, `p2Cw`, `r9Nv`). Indices 0=female and 1=male are
   derivable with certainty from the filter logic (§2.3). **Index 2 is unknown**
   — it is never selected by any shipping setting. Index 3 (`r9Nv` = "person")
   belongs to a model variant not shipped in this build.
5. **No head-to-head benchmark was run.** All latency numbers below are each
   project's own constants and self-reported thresholds, not measurements by us.
   §4 makes measuring this the first deliverable.

---

## Part 1 — Porda-AI: exactly what it does

### 1.1 The model

A single Darknet-format YOLO, bundled in-repo.

| Property | Value | Evidence |
| --- | --- | --- |
| Architecture | YOLOv4-tiny variant, 2 detection heads (stride 16 + 32), CSP-style group-routed blocks | `model/pordav4x3.cfg:25-189` |
| Input | **544 × 320** (non-square, landscape — matched to browser windows) | `model/pordav4x3.cfg:8-9` |
| Classes | **2** | `model/pordav4x3.cfg:220, 269` |
| Class semantics | `0 = male`, `1 = female` | `main.py:279-282` (`is_detect_male` → append 0, `is_detect_female` → append 1) |
| Head width | `filters=21` = 3 anchors × (5 + 2 classes) | `model/pordav4x3.cfg:212, 263` |
| Anchors | `10,14  23,27  37,58  81,82  135,169  344,319` | `model/pordav4x3.cfg:219, 268` |
| NMS (training cfg) | `nms_kind=greedynms`, `beta_nms=0.6`, `scale_x_y=1.05`, `iou_loss=ciou` | `model/pordav4x3.cfg:223-233` |
| Weights | `model/porda-19200-lr-0005-909.weights`, 24 MB — 19 200 iterations, lr 5e-4, **mAP ≈ 90.9** (the `909` suffix; prior candidates `906`, `895`, `889` are commented out) | `SetupPordaApp.py:210-217` |

**Weight loading** (the brief assumed they were absent — they are not):
`cv2.dnn.readNetFromDarknet(CONFIG, WEIGHTS)` at `main.py:174-175`, with paths
resolved in `SetupPordaApp.py:197-217`. Before falling back to the bundled pair,
the app scans `%APPDATA%/PordaAI/Extarnal-Model` for any `.cfg`/`.weights` pair
and prefers it (`SetupPordaApp.py:180-203`) — **a user-swappable model slot**, the
same idea as our model packs, shipped years earlier and with no hash check.

### 1.2 Preprocessing, thresholds, postprocessing

```python
# main.py:603-606
padded_frame, x_ratio, y_ratio = self.add_padding(frame, h, w)
classes, scores, boxes = self.model.detect(
    padded_frame, confThreshold=self.detection_accuracy, nmsThreshold=0.1)
```

- Letterbox is **top-left anchored, black fill, bottom/right padding only**
  (`main.py:782-801`), and is **skipped entirely** when residual padding is small
  (`bottom < 55 and right == 0`, or `right < 70 and bottom == 0`,
  `main.py:791-792`) — a ~2 % geometry error traded for a `cv2.resize` call.
- Normalisation `scale=1/255`, no mean subtraction, `setInputParams` in
  `settings/EngineSetting.py:46, 68, 122`.
- **Confidence threshold 0.25** (`accuracy: 25` / 100, `settings/SettingsValue.py:8`
  → `main.py:260`). User-settable 1–99 (`settings/Settings.py:513`).
- **NMS IoU 0.10** — extremely aggressive. Two overlapping people merge into one
  box. For a censoring product that is the *safe* direction: over-merge, never
  under-cover.
- Network size is user-tunable in units of 32: width 5–19 (160–608), height 5–15
  (160–480) (`settings/Settings.py:569, 580`). Locked to the default once a GPU
  engine is selected (`main.py:244-247`).

### 1.3 Per-frame pipeline

| Stage | Behaviour | Evidence |
| --- | --- | --- |
| Capture | Win32 `PrintWindow` of the **foreground window's client rect**, full native resolution, BGRA→BGR | `GetDataV2.py:162-194` |
| Scope filter | Only if the owning process is in the include-list; default `chrome, msedge, brave, firefox, opera, PotPlayerMini64, vlc` | `settings/SettingsValue.py:41-43`, `GetDataV2.py:98-143` |
| Sticky focus | If the new foreground window is < 650 × 400 px (popups, the settings dialog), **revert to the last interesting hwnd** and keep detecting on it | `GetDataV2.py:165-173` |
| Backend | OpenCV DNN. `DNN_BACKEND_DEFAULT` + `DNN_TARGET_OPENCL`; CPU otherwise. **CUDA explicitly unsupported.** Writes `GpuPreference=1` into the DirectX registry to force the dGPU | `settings/EngineSetting.py:63-68, 95-97, 114-122, 173-207` |
| Cadence | `QTimer`, two-state: **active 65 ms (~15 fps)**, **sleep 500 ms (2 fps)** | `settings/SettingsValue.py:11-13` |
| State machine | Any target-class detection resets `last_object_found_time` and, after `keep_running_timeout = 10 s` without one, drops to sleep. Capture failure drops to sleep within 1 s | `main.py:608-643`, `main.py:530-539` |
| Concurrency | **None.** Single-threaded on the Qt main thread, one `detect()` per tick, no queue, no cache | `main.py:496-526` |
| Memory | `cv2.dnn.DNN_CLEAR_FORWARD/BACKWARD = 1` (cargo-cult: these are not real OpenCV flags); DC/bitmap handles released per frame | `settings/EngineSetting.py:44-45`, `GetDataV2.py:189-192` |
| Thermal guard | Optional watchdog thread averages `psutil.cpu_percent` over 30 s and **auto-deactivates** above `max_cpu_limit` (default 90, feature off by default), reactivating when it drops | `main.py:74-97, 463-491` |
| Telemetry | GA4 ping every 15 000 frames | `main.py:514-520`, `settings/ga4_porda.py` |

### 1.4 Decision policy and rendering

```python
# main.py:647-669
for class_id, box, score in zip(classes, boxes, scores):
    if class_id in self.object_list:        # the entire policy
        ...
        cover_img = cv2.blur(image, (w//2, w//2))
```

That is the whole policy. **No thresholds per class, no second model, no NSFW
pass, no face or gender stage, no pose, no clothing reasoning.** The detector's
class *is* the verdict. Default `object_list = [1]` — women only
(`settings/SettingsValue.py:25-30`).

Rendering is a frameless, always-on-top, `WA_TranslucentBackground`,
`WindowTransparentForInput` Qt overlay (`main.py:102-111`) painting per-box
pixmaps (`main.py:803-811`). Three cover styles:

- **Blur** (default): `cv2.blur` with kernel `w // 2` — i.e. a box blur whose
  radius scales with the detection, so a large subject is destroyed as thoroughly
  as a small one (`main.py:665-669`).
- **Solid colour**, default `(0,0,255)` (`main.py:661-663`).
- **Dominant background colour**: samples a `60×50` strip inside the box top,
  `np.unique(..., return_counts=True)`, fills with the modal colour
  (`main.py:655-659`). This is what the README calls "Optimal Halal Vision" —
  the cover blends into the page instead of announcing itself.

**Occlusion subtraction** (`main.py:673-773`): the app tracks the rects of other
top-level windows and, when a detection box intersects one, paints only the
*non-intersecting* sub-rectangles in black. This is the "no cover-bleed when
another window sits on top" release note. ~100 lines of rectangle algebra, and
it is the single most-polished piece of code in the repo.

**No tracking. No per-box hysteresis. No detection cache.** Flicker is controlled
only by the 65 ms cadence and a periodic `SetWindowPos(HWND_TOPMOST)`.

### 1.5 Why it handles fitness content well, architecturally

Because *clothing state is not part of its model of the world.* Its two classes
are `male` and `female` — whole figures. A woman in a sports bra, a woman in an
abaya, and a woman in a parka produce the same detection and the same cover. At
confidence 0.25 it fires readily; at NMS 0.10 it merges rather than splits. It
cannot *under*-cover an immodestly dressed woman, because it was never asking
about dress. Its failure mode is the opposite: it covers everyone it recognises.

### 1.6 Worth stealing

1. **The two-state active/sleep cadence with a hold-down timer** (`keep_running_timeout`)
   — richer than our pure exponential backoff: it holds full rate for 10 s after
   the *last* positive rather than decaying from the last *sample*.
2. **Sticky focus** — keep working on the last interesting surface when focus
   moves to a small transient window. Our analogue: don't drop a video's sampler
   when a dialog/overlay steals observation.
3. **Dominant-background-colour cover.** Visually far less alarming than a blur
   smear, and trivially cheap. We have `useSolidColor`-equivalent machinery; a
   "match the page" mode is a few lines.
4. **Blur kernel scaled to box size** rather than a fixed px radius. Our fixed
   `blurAmount: 20` under-blurs a 600 px subject and over-blurs a 40 px one.
5. **User-swappable external model directory** — validates our pack system, and
   suggests the directory-scan UX is simpler than our manifest approach.
6. **The CPU watchdog** that disables detection rather than degrading it. Honest.

---

## Part 2 — Upstream HaramBlur 0.6.11: exactly what it does

### 2.1 The detector: three sizes of one proprietary YOLOv8

```js
// dist/offscreen.js:9365 (byte 2012161)
var Pd = {
  vZNq2WHrFm7b: { inputShape:[1,640,640,3], classNames:["x7kQ","m3Jz","p2Cw"],
    config:{ cacheSensitivity:.5, iouThreshold:.7, scoreThreshold:.45,
             maxDetected:70, skipFrames:20, skipTime:1e3 },
    binFiles:{ "group1-shard1of3.bin":"560eabac6eb0.bin", ... } },
  mcDNsuzfxMHy: { inputShape:[1,416,416,3], /* identical config + classes */ },
  uSmhjQtBQ35f: { inputShape:[1,320,320,3], /* identical config + classes */ } }
```

Three input sizes, **same three classes, same thresholds**. The decode path
pins the architecture beyond doubt:

```js
// dist/offscreen.js:9365 (byte ~3205600), Lk.detect()
o = this.model.execute(f);
s = o.transpose([0,2,1]);                     // [1, 4+nc, N] -> [1, N, 4+nc]
i = cxcywh -> xyxy from channels 0..3;
[u,l] = s.slice([0,0,4],[-1,-1,nc]).squeeze(0) -> .max(1), .argMax(1);
a = tf.image.nonMaxSuppressionAsync(i, u, this.maxDetections /*70*/,
                                    this.iouThreshold /*0.7*/, n /*score*/);
```

Channel-major `[1, 4+nc, 8400]`, no objectness channel, anchor-free,
transpose-then-argmax — that is the **YOLOv8 detection head**, exported to a
TFJS graph model. Same family as our NudeNet `320n.onnx`
(`docs/ENGINE_NOTES.md:7`). Letterboxing is **centre-padded** with `/255`
normalisation (`Nae`, byte ~3201859) — unlike Porda's top-left pad and unlike
our top-left `rgb(114,114,114)` pad.

### 2.2 How weights are loaded (and "protected")

- 9 shard files in `src/mdls/*.bin` (3 per model, 2–4.2 MB each), named by hash.
- Each shard is **XOR'd with a repeating 32-byte key** assembled at runtime from
  four hard-coded hex literals (`dist/offscreen.js:9365`, byte 2010619):
  `"3561f25a5be3574f" + "d4d9b52e48bb9c23" + "87a16e40d62cff46" + "f5c43a4f1643e5d3"`.
- `model.json` is **AES-CBC** under the same key, IV from
  `"d2ac637263e93efc" + "a3f0aa736d252ff1"`, base64, prefix-delimited by `:`.
- Anti-debugging: `UMe()` (byte 2011796) installs `setInterval(() => { debugger }, 100)`,
  a 500 ms timing trap, and wraps every `window.console` method to re-trigger it.

Key and ciphertext ship in the same bundle, so this is obfuscation, not
encryption. **We did not extract the weights** (§0, gap 3). Noting the scheme
matters for one reason only: *their moat is not cryptographic, it is the
training set.* Which is the correct read of where the real gap lies.

### 2.3 Class semantics

```js
// dist/offscreen.js:9365 (byte 3201740)
YMe = (det, male, female) => { let n = det?.classIndex,
  o = n===0, s = n===1, a = o||s||n===2||n===3,
  u = (male?1:0)+(female?1:0);
  return u ? (u===2 ? s||o : (male ? s : o)) : a; }
```

With only `blurFemale` on (the default), `u===1`, `male===false`, so the filter
returns `o`, i.e. `classIndex === 0`. Therefore **class 0 = female, class 1 = male**.
Class 2 passes only when *neither* gender toggle is on — a state in which nothing
is blurred anyway. **Class 2's meaning is unrecoverable from this build.**

`_P = { FULL:["x7kQ","m3Jz","p2Cw","r9Nv"], TFJS:["x7kQ","m3Jz","p2Cw"] }` plus
`personLabel: "r9Nv"` (byte 3193068, 3195302) shows a 4-class "person" variant
exists upstream of the TFJS export, used as the generic label for a *dormant*
two-stage path (§2.8).

### 2.4 The "ensemble": not an ensemble

This is the most-misunderstood part of upstream, including in our own
`docs/DETECTION_PARITY.md:20`. There is **no voting and no face/gender model**.

**The Human library is loaded with every feature disabled:**

```js
// dist/offscreen.js:9365 (byte 3193680)
XMe = { modelBasePath: chrome.runtime.getURL("src/models/human"), backend:"",
  cacheSensitivity:.9, warmup:"none", async:!0, filter:{enabled:!1},
  face:{ enabled:!1, iris:{enabled:!1}, mesh:{enabled:!1}, emotion:{enabled:!1},
         detector:{enabled:!1}, description:{enabled:!1} },
  body:{enabled:!1}, hand:{enabled:!1}, gesture:{enabled:!1}, object:{enabled:!1} }
```

Corroborating: `src/models/human/` **does not exist in the CRX** (the package
ships `src/mdls/` only). Human 0.6.11 is a TFJS bundle and backend-negotiation
shim — nothing more. BlazeFace, FaceMesh, SSR-Net age, FaceRes gender, emotion
and anti-spoof are all dead code in the 4.5 MB bundle.

**What actually runs** is a two-model cascade with a hard gate:

```js
// dist/offscreen.js:9365 (byte 3222384)  Wk = runDetection(img, isVideoFrame)
g = isVideoFrame ? strictness*0.3 + 0.17 : strictness;   // NSFW strictness
y = settings.getObjectScoreThreshold();                   // detector threshold
if (specificBlur) {
  S = await Et.detectGender(t, blurFemale, blurMale, y, skipCache);
  if (!S?.length) return { label:false };                 // <<< HARD GATE
  E = await nsfw.nsfwModelClassify(t, skipCache);
  return oF(E,g)?.nsfw ? {label:"nsfw", body:S} : {label:"face", body:S};
}
T = await nsfw.nsfwModelClassify(t, skipCache);            // whole-blur path
if (oF(T,g)?.nsfw) return { label:"nsfw", body:null };
S = await Et.detectGender(...); return S?.length ? {label:"face", body:S} : {label:false};
```

- In **region mode (the default)** the object detector is the sole gate. No
  female box → `{label:false}` → **nothing is blurred, and NSFWJS is never
  consulted.** NSFWJS only *renames* the verdict `face` → `nsfw`; the regions are
  the detector's boxes either way.
- In **whole mode** NSFWJS short-circuits first; the detector is the fallback.
- The two models therefore form a **gate + escalation label**, not a vote. Our
  `docs/DETECTION_PARITY.md:20` line "proprietary 3-class + Human face/gender +
  nsfwjs enrich" is wrong on two of three terms.

### 2.5 NSFWJS: remote-hosted, margin-scored

| Property | Value | Evidence |
| --- | --- | --- |
| Source | **`https://hb-nsfw-model.pages.dev/model.json`** fetched at runtime | byte 3193560 |
| Architecture | NSFWJS MobileNet, 5 classes `Drawing, Hentai, Neutral, Porn, Sexy` | byte 3194242 |
| Input | 224², `/255`, centre-letterboxed, `topK = 3` | `ZMe`, byte 3194160 |
| Temporal skip | `skipTime 4000 ms`, `skipFrames 50`, `cacheSensitivity 0.9` | `ZMe` |

Thresholds are a **function of strictness**, clamped to `[0.1, 0.6]`:

```js
// dist/offscreen.js:9365 (byte 3194242)
Tae = (s=0) => { let e = clamp(s, .1, .6); return {
  0:{Drawing, nsfw:false, thresh:.5},        1:{Hentai, nsfw:true,  thresh:.5+(1-e)*.7},
  2:{Neutral, nsfw:false, thresh:.5+e*.5},   3:{Porn,   nsfw:true,  thresh:.1+(1-e)*.5},
  4:{Sexy,    nsfw:true,  thresh:.1+(1-e)*.6} }; }

oF = (preds, s) => { /* t = max over NSFW classes of (p - thresh)
                        n = max over SFW  classes of (p - thresh) */
  return { nsfw: (t - n) >= 0.15, reallyNSFW: Porn>.7 || Sexy>.9 || Hentai>.8 }; }
```

At default strictness 0.4: Hentai 0.92, Neutral 0.70, Porn 0.40, Sexy 0.46.
The verdict is a **margin** (`best-NSFW-excess − best-SFW-excess ≥ 0.15`), not a
bare threshold — strictly better calibrated than a fixed cut, and conceptually
the same idea as our `nsfwThreshold` band (`src/shared/policy.ts:156-158`) but
with per-class strictness-dependent offsets we don't have. `reallyNSFW` is
computed and **never read** — dead code.

Video frames get compressed strictness `s*0.3 + 0.17` → **0.29** at default, i.e.
video is held to a *more sensitive* NSFW bar than images.

### 2.6 Strictness → detector threshold: a 0.08-wide band

```js
// dist/offscreen.js:9365 (byte ~3209900)
iF=.1, B1=fI.strictness /*0.4*/, Dae=1, JMe=.48, Pae=.45, QMe=.40;
getObjectScoreThreshold() {
  let t = clamp(Number(strictness), 0.1, 1);
  return t <= 0.4 ? lerp(0.48, 0.45, (t-0.1)/0.3)
                  : lerp(0.45, 0.40, (t-0.4)/0.6); }
```

The entire strictness slider moves the detector confidence from **0.48 to 0.40**.
Default **0.45**. This refutes `docs/DETECTION_PARITY.md:21` ("per-class
thresholds scaled down to ~0.1 floor") — the ~0.1 floors belong to the *NSFWJS*
table, not the detector. Upstream's detector threshold band is **narrower than
ours** (our face curve spans 0.6→0.2, `src/shared/policy.ts:97`). They do not
win on threshold aggressiveness; they win on what the classes mean.

### 2.7 Backend, model selection, cadence, concurrency

| Property | Value | Evidence |
| --- | --- | --- |
| Backend ladder | `webgpu → humangl(WebGL) → wasm → cpu`, first success wins, sticky in `localStorage` for **3 days** | `kae` byte 3193560; `Mk.initHumanTF` byte 3197264 |
| WASM source | local `tfjs/` SIMD → local non-SIMD → jsDelivr CDN `@tensorflow/tfjs-backend-wasm@4.17.0` | `Mk.initHumanTF` |
| Backend constraints | `wasm` → pinned to the 320 model; `cpu` → `skipObjectDetection: true`, **NSFW-only mode**, region blur force-disabled | `hI` byte ~1302400; init byte 3233198 |
| Model auto-select | On-device benchmark: 2 warm-up + up to 30 inferences on a synthetic 1024² tensor, IQR outlier trim. **≤60 ms → 640; ≤100 ms → 416; else 320.** Early-exit after 10 samples if avg > 150 ms; 5 s wall timeout → 320. Cached 3 days | `yI.runBenchmark` byte ~3190700; `Lh` |
| Image concurrency | `maxProcessing = 3` concurrent inferences, `maxLoading = 50` concurrent decodes | `sF` byte 3207981 |
| Min image size | 30 × 30 px | `zTe/BTe = 30`, `content.js:163` |
| Video cadence | **320 → 15 fps, 416 → 20 fps, 640 → 25 fps** (faster model ⇒ *higher* target rate) | `Se()`, `content.js:162` (byte 17075) |
| Video loop | `requestVideoFrameCallback` with rAF fallback; strict one-in-flight (`activeFrame`); results older than **0.5 s** discarded; element abandoned after 10 errors | `content.js:1` (byte ~27282) |
| Temporal frame-skip | L2 pixel diff vs previous input tensor ÷ (h·w·255·3) ≤ `cacheSensitivity`; reuse prior detections for up to `skipFrames 20` frames / `skipTime 1000 ms`. **Images bypass this; video uses it.** | `Lk.objectModelSkip` byte 3202914 |
| Verdict cache | IndexedDB `HaramBlurCache/detections`, 1 000 entries, 32 MB, cleared on version change | byte ~3192900 |
| Transport | Long-lived `chrome.runtime.connect` ports `hb-img-det-port` / `hb-vid-det-port`, exponential reconnect `1 s · 2ⁿ` capped at 30 s | `content.js:162-163` |
| CORS fallback | `fetch → FileReader → data: URL`, with a `sessionStorage` per-host failure counter that gives up after 5 | `pt`, `content.js:163` |
| Self-heal | Hourly self-test inference on the bundled icon; `recreateOffscreen` on failure. GPU OOM detection parses shader error text, halves `textureLimit`, clears the benchmark | byte ~3233900; `sg` constructor byte ~3194900 |

### 2.8 Dormant second stage — their roadmap, visible in the bundle

`nF` (byte 3195302) is a crop-classifier class with `personLabel: "r9Nv"` and
`inputShape [1,224,224,3]`; `Lk.enrichDetectionsWithClassification` re-labels
each detection's `class`/`gender`/`genderScore` from it. **`classifier` is null
for all three shipped models** (`$P` spreads `Pd`, which has no `classifier`
key), so the path never executes. They are building toward *person-detect →
per-crop gender classifier* — which is exactly the inversion our reviews
proposed, arrived at independently from the other direction.

### 2.9 Strictness/sensitivity semantics and blur UX

Defaults (`fI`, `dist/offscreen.js:5001` byte 1301023; mirrored `content.js:1`):

```js
status:true, blurryStartMode:false, blurAmount:30, blurImages:true, blurVideos:true,
blurMale:false, blurFemale:true, specificBlur:true, unblurImages:false,
unblurVideos:false, gray:false, useSolidColor:false, solidColor:"#808080",
strictness:.4, scoreThreshold:.4, detectionModel:"vZNq2WHrFm7b" /* 640 */
```

Two distinct blur renderers:

1. **Baked image replacement (images only).** The offscreen document draws the
   original to a canvas and composites blurred crops, then returns a
   `toDataURL()` and the content script **swaps `img.src`**
   (`TY`, `dist/offscreen.js:5001` byte 1305895; `Lt`, `content.js:163`).
   Per box: expand by `pad = max(bw,bh)/20`, apply CSS
   `filter: blur(r)[ grayscale(100%)]` **twice**, composite back.
   `r = max(W,H)/20 · blurAmount/50` — **scaled to image size, like Porda, unlike us.**
   Images under 70 px force whole-image blur.
2. **DOM portals (video, and image fallback).** `.hb-wrapper` absolutely
   positioned sibling with `.hb-rect` children using `backdrop-filter: blur()`
   (`j`, `content.js:163` byte 21150) — architecturally identical to our
   `#hb-region-portal`. **Overlapping rects are merged by bounding-box union
   before painting** (same function), which we do not do.
   Whole-element fallback is `video.hb-blur { filter: blur(30px) }`.

Other UX:

- **`blurryStartMode`** (off by default): CSS animations `hb-blur-temp` 20 000 ms
  and `hb-hide-temp` 6 000 ms hold new media blurred/hidden until a verdict
  arrives (`content.js:1` byte ~7999). **The zero-leak primitive the reviews ask
  for already exists upstream — shipped, and off by default.**
- **Video hysteresis:** `kt = 1` positive frame → blur; `He = 2` consecutive
  clear frames → unblur (`content.js:162`, consumed in `De`, byte ~29700).
  Asymmetric and safety-biased. Ours is symmetric at 2/2
  (`src/content/video/sampler.ts:32, 98, 105`).
- `gray`, `useSolidColor` (`brightness(0)` + solid background), per-video
  draggable on/off controller widget, site whitelist.

### 2.10 Gender/age, and covered-vs-exposed

- **Age: not handled at all.** No age model, no child skip. Ours has
  `skipChildren` with `CHILD_AGE_MAX = 15` (`src/shared/policy.ts:127-137`) —
  a genuine feature lead.
- **Gender: a detector class, not a classifier.** No face crop, no alignment, no
  landmark stage. Cheaper, and immune to the "small thumbnail face" failure mode
  that `docs/reviews/dp6-eval-opus-5.md:29-31` warns about in our design.
- **Covered vs exposed: the distinction does not exist.** Nothing in the bundle
  reasons about clothing, skin, or coverage. A woman in a sports bra and a woman
  in a parka are both `classIndex 0`, both blurred, with the same box. The only
  explicitness signal is whole-image NSFWJS, which cannot localise and (in region
  mode) cannot even fire unless the detector already found a person.

**This is the finding that reframes our whole problem.** Our core gap —
"FEMALE_BREAST_COVERED is not blur-capable, BELLY_EXPOSED is modesty-gated" — is
a gap in a question **neither competitor asks**. They do not distinguish covered
from exposed; they blur the woman.

---

## Part 3 — Honest comparison

### 3.1 Corrections to `docs/DETECTION_PARITY.md:14-22`

| Claim there | Reality | Evidence |
| --- | --- | --- |
| "upstream: proprietary 3-class **+ Human face/gender** + nsfwjs enrich" | Human runs with **every model disabled**; there is no face or gender model. NSFWJS is a gate/label, not an enrichment vote | §2.4 |
| "upstream: per-class thresholds scaled down to **~0.1 floor**" | That's the NSFWJS table. The **detector** band is 0.48→0.40 — narrower than ours | §2.6 |
| "Porda-AI 1.1.x: detector input **320×320**" | **544 × 320**, user-tunable 160–608 × 160–480 | `model/pordav4x3.cfg:8-9` |

### 3.2 Capability matrix

Legend: **🔴 behind** · **🟡 even** · **🟢 ahead** (ours, vs the better competitor on that row).

| Capability | Porda-AI | Upstream HB 0.6.11 | Ours | |
| --- | --- | --- | --- | :-: |
| **Detection schema** | 2 classes: male, female (whole figure) | 3 classes: female, male, unknown (whole figure) | 18 NudeNet body-part classes + 2 NSFW + 1 synthetic face | 🔴 |
| **What the class answers** | "Is a woman here?" | "Is a woman here?" | "Is this specific part exposed?" | 🔴 |
| **Covered-vs-exposed reasoning** | None — irrelevant by design | None — irrelevant by design | Explicit, and **miscalibrated**: `FEMALE_BREAST_COVERED` fires 0.7–0.83 on positives *and* negatives; `BELLY_EXPOSED` gated behind `strictModesty: false` | 🔴 |
| **Detector input** | 544 × 320 (160–608 × 160–480) | 640² / 416² / 320², auto-selected by on-device benchmark | 320² WASM / 480² WebGPU; optional 640m pack | 🔴 |
| **Input selection** | Manual spinbox | **Benchmarked, cached 3 days, early-exit** | Static per backend | 🔴 |
| **Backend** | OpenCV DNN + OpenCL (no CUDA) | TFJS: WebGPU → WebGL → WASM → CPU, sticky 3 days | ORT: WebGPU → WASM, **per-model** | 🟢 |
| **Degraded-backend policy** | CPU fallback, same model | CPU → NSFW-only, region blur disabled; WASM → 320 pinned | Per-model fallback, no feature downgrade | 🟡 |
| **Models in the pipeline** | 1 | 2 (detector + NSFWJS) | **6** (NudeNet, YuNet, FastFace, NSFW, YOLOv8n-seg, + gender/age head) | 🔴 complexity |
| **Model provenance** | Bundled, user-swappable dir, no hash check | **Detector bundled (XOR+AES), NSFWJS fetched from the network at runtime** | All bundled, hash-verified, zero-network | 🟢 |
| **Confidence threshold** | 0.25 flat | 0.45 (slider 0.48→0.40) | Per-class `{max,min,pow}` curves, 0.6→0.12 depending on class (`src/shared/policy.ts:97-124`) | 🟢 *(mechanism)* 🔴 *(outcome)* |
| **NSFW whole-image** | ✖ | NSFWJS 5-class, margin ≥ 0.15, strictness-dependent per-class thresholds; **video held to a stricter bar** | NSFWJS 5-class → `NSFW_EXPLICIT/SUGGESTIVE` full-frame detection, single threshold band 0.4→0.15 | 🟡 |
| **Face/gender stage** | ✖ | ✖ (dead code present) | YuNet 640² + FastFace 112² aligned crops, relabel gate 0.7/0.85 → `FACE_PERSON` abstention | 🟢 |
| **Age / child skip** | ✖ | ✖ | `skipChildren`, `CHILD_AGE_MAX = 15` | 🟢 |
| **Person instance segmentation** | ✖ | ✖ (4th "person" class exists upstream, not shipped) | YOLOv8n-seg 640², masks, face↔person association | 🟢 *(unused for the decision)* |
| **NMS** | greedy, **IoU 0.10** (aggressive merge) | `tf.image.nonMaxSuppressionAsync`, IoU 0.70, maxDet 70 | per-class greedy, IoU 0.45 (`src/engine/nudenet.ts:74`) | 🟡 |
| **Temporal frame-skip** | ✖ | **L2 pixel-diff skip**, 20 frames / 1 s (detector), 50 / 4 s (NSFW) | Scene-change detection feeds backoff only | 🔴 |
| **Tracking / instance identity** | ✖ | ✖ | ✖ | 🟡 *(nobody has it)* |
| **Flicker control** | Cadence only | 1 frame on / 2 frames off (asymmetric) | 2 on / 2 off (`sampler.ts:32`) | 🟡 |
| **Video cadence** | 15 fps active / 2 fps sleep, 10 s hold-down | 15–25 fps, model-dependent, one-in-flight | one-in-flight, 10 fps ceiling → 1 fps backoff; measured ~7.5 fps | 🔴 |
| **Scope of capture** | Foreground window only, full native res | Per `<img>`/`<video>`, downscaled to model size | Per element, capped at detector edge | 🟡 |
| **Image concurrency** | n/a (1 window) | 3 inferences / 50 decodes | queue in `src/content/queue.ts` | 🟡 |
| **Verdict cache** | ✖ | IndexedDB, 1 000 entries / 32 MB, version-invalidated | LRU in-engine (`src/engine/cache.ts`) | 🟡 |
| **Region blur mechanism** | Per-box pixmap overlay (OS-level) | Baked `toDataURL` src-swap (images) + `backdrop-filter` portals (video) | `backdrop-filter` portals + shadow DOM | 🟡 |
| **Region geometry** | Whole detected figure | Whole detected figure | `coverageMode: "head"` **by default** — face box ×1.7 w ×1.9 h (`src/content/region.ts:87-89`) | 🔴 |
| **Overlapping region merge** | ✖ (NMS 0.1 does it upstream) | ✔ bounding-box union before paint | ✖ | 🔴 |
| **Blur radius scaling** | `kernel = w/2` (box-relative) | `max(W,H)/20 · amount/50` (image-relative), applied twice | Fixed `blurAmount: 20` px | 🔴 |
| **Escalation to whole-element** | ✖ | `isSpecificBlur` off, `forceWholeBlur` backend constraint, images < 70 px | `smartCensor`: core-exposed labels, ≥2 female faces, or regions > 30 % area (`src/content/region.ts:200-204`) | 🟢 |
| **Occlusion handling** | ✔ window-rect subtraction | n/a (DOM) | n/a (DOM) | 🟡 |
| **Zero-leak / blur-first** | ✖ | **`blurryStartMode` shipped, default off** | ✖ | 🔴 |
| **Reveal UX** | ✖ | hover-unblur per setting, per-video toggle widget | hover / hold-to-peek / off, context menu | 🟢 |
| **Telemetry / network** | GA4 every 15 k frames | Firebase auth, Google OAuth, uninstall diagnostics, hourly blocklist fetch, runtime model fetch | **None** | 🟢 |
| **Self-heal** | CPU watchdog auto-disable | Hourly self-test, GPU-OOM texture-limit halving, offscreen recreation | boot/infer timeouts, backend fallback | 🟡 |
| **Licence** | AGPL-3.0 app, weights bundled unrestricted | MIT-ish shell, **proprietary weights, RE forbidden** | AGPL-3.0, all weights open | 🟢 |
| **Measured perf** | Unknown (no numbers in repo) | Implied device tiers: 640 ≤ 60 ms, 416 ≤ 100 ms, 320 otherwise | 85 ms/image WASM (`docs/ENGINE_NOTES.md:3`); video ~7.5 fps | 🔴 *(unmeasured head-to-head)* |
| **Eval rigour** | mAP 90.9 in the weights filename; no eval set in repo | None visible | 94.1 % acc / 100 % prec on **n = 17**, distribution-mismatched | 🔴 |

### 3.3 The one-paragraph verdict

We are **ahead on engineering** — open weights, zero network, per-model backend
selection, face/gender/age, instance segmentation, escalation policy, richer
threshold mechanics, better reveal UX, real licence hygiene. We are **behind on
the only thing that decides whether the product works**: what the detector's
classes mean, and what geometry we blur. Both competitors answer "is a woman
present" with one whole-figure box at 0.25–0.45 confidence and censor that box.
We answer "which body part is exposed" with eighteen classes, then censor a
**head-sized box** derived from a *face* detection. On a YouTube fitness
thumbnail both of those are the wrong answer in the same frame: the head blurs,
the sports bra and midriff do not, and the user correctly reads that as the
product being broken.

---

## Part 4 — The pivot plan

### 4.0 What the competitor evidence does to the prior reviews' proposals

| Prior proposal | Verdict against competitor evidence | Why |
| --- | --- | --- |
| **Person-instance-first pipeline inversion** (`dp6-eval-opus-5.md:70-86`) | **Validated in direction, over-scoped in means** | Both competitors are person-figure-first. *Neither* uses segmentation, masks, or association. They get there with one detector whose classes already encode the gender decision. We already produce the same information (NudeNet `FACE_FEMALE` + YuNet/FastFace gender + YOLOv8n-seg instances) — we just don't route the blur region from it. The inversion is a **policy/region change**, not a new pipeline. |
| **Self-calibrated skin fraction** (`dp6-eval-opus-5.md:87-96`) | **Refuted as a parity requirement; retained as a surpass move** | Neither competitor computes coverage, and both beat us on the target content. Coverage is not what's missing for parity. It becomes valuable *after* the inversion, as the FP control that lets us blur *less* than they do (a fully-dressed woman in a news photo) without losing the fitness case — i.e. it's our differentiator, not our catch-up. |
| **Clothes-parsing segmentation** (SegFormer/ATR) | **Deprioritised** | Same argument, plus 15–40 ms/frame and a licence minefield on the checkpoints. Correct as an opt-in accuracy pack (DP3 §3 infrastructure), wrong as the parity path. |
| **CLIP prompt arbitration** | **Deferred, but strategically right** | No competitor has semantic arbitration. It is the only proposal that makes *modesty definable as editable data* — structurally impossible for proprietary weights. But it answers a question we haven't earned the right to ask until regions come from person instances. |
| **Temporal tracking** (both reviews) | **Validated and under-rated** | Confirmed: **neither competitor tracks.** Upstream's entire temporal state is a 1-on/2-off frame counter and an L2 pixel-diff skip. Porda has none. Track-level evidence accumulation is an uncontested lead available for ~30 lines of CPU logic. |
| **Zero-leak mode** (both reviews) | **Validated, and urgent — upstream already ships the primitive** | `blurryStartMode` exists in 0.6.11 (§2.9), default off. If they flip that default we lose the differentiator before we build it. |
| **Real stratified eval set** (both reviews, both ranked it top-3) | **Validated — and competitor evidence makes it sharper** | Porda publishes its mAP in a filename; upstream ships no eval at all. Neither has an eval advantage. n=17 with a distribution that excludes the target content is *why* we shipped this gap, exactly as both reviews said. |
| **Crop-classifier fine-tune** (`dp6-eval-opus-5.md:103`, `opus-4-8.md:44`) | **Validated, and the competitor bundle proves the shape** | Upstream's dormant `nF` classifier with `personLabel:"r9Nv"` (§2.8) is *literally* a person-detect → crop-classifier design, shipped-but-disabled. They are building the thing the reviews recommend. The moat is the training set, not the architecture — §2.2 shows their "protection" is XOR. |

**The implication of "neither competitor uses person-instance-first, and both
beat us":** person-instance-first is not what makes them good. **Whole-figure
region geometry plus a gender-level class taxonomy** is what makes them good.
Person-instance segmentation is one way to get there; we happen to already own
that way, so we should use it. But if instance segmentation were deleted
tomorrow we could still reach parity, because the gap is region selection and
class semantics — not localisation quality.

### 4.1 Move 0 — Measure before writing any detection code

**Nothing below is justified until these three numbers exist.** All are
measurable with existing tooling (`scripts/probe-dets.ts`, `scripts/eval.ts`,
`tests/fixtures/yt-*.jpg`).

| # | Measurement | Why it gates the plan |
| --- | --- | --- |
| **M1** | On the 6 YouTube fitness fixtures and every dev-split image containing a woman: does a `FACE_FEMALE` detection at ≥ `faceThreshold(0.5)` = 0.4 exist, and what fraction of the subject's visible body does the current `coverageMode:"head"` box cover? | If `FACE_FEMALE` fires but the head box covers < 25 % of the figure, the gap is **region geometry**, and §4.2 is a config change, not a project. This is the single highest-value number in the report. |
| **M2** | Does YOLOv8n-seg produce a person instance on each of those fixtures, and does `associateFacesToPersons` bind the female face to it? | Determines whether §4.3 can use masks or must use the `coverageMode:"body"` ratio box. |
| **M3** | On dev-split *negatives* that currently stay clear: how many would blur if any confident `FACE_FEMALE`/person-instance blurred the whole figure? | Prices the FP cost of competitor-equivalent behaviour. Both competitors accept a high FP rate here. We need to know ours before we choose to. |

Plus the anchor-rate measurement `dp6-eval-opus-5.md:13` already demanded, which
M1 subsumes.

**Size:** 1–2 days. **Acceptance:** a table in a PR description, no code changes.

---

### 4.2 Move 1 — Region geometry parity *(the highest-leverage first move)*

**Competitor evidence.** Porda blurs `boxes` straight from the 2-class detector
(`main.py:647-679`) — the whole figure. Upstream blurs `r.body[].bbox` straight
from the 3-class detector (`TY`, `dist/offscreen.js:5001`; `j`, `content.js:163`)
— the whole figure. Both blur radii scale with the region
(`main.py:668`, `TY` `g = max(W,H)/20·amount/50`). Neither has anything like our
`coverageMode:"head"`.

**The change.**

1. Default `coverageMode` from `"head"` → `"body"`
   (`src/shared/settings.ts:135`). `coverageBoxFor` already implements it:
   2.4× face width, crown to 4.5 face-heights below the chin
   (`src/content/region.ts:91-94`), with the person mask substituted when
   available (`src/content/region.ts:138-141`).
2. Scale `blurAmount` by region size instead of using a fixed px radius, matching
   both competitors.
3. Merge overlapping regions by bounding-box union before painting (upstream's
   `j` does this; we don't) — avoids double-`backdrop-filter` seams on adjacent
   subjects.
4. Re-tune `escalateAreaPct` (default 0.30, `src/shared/settings.ts:137`). With
   body coverage, a single 9:16 subject will routinely exceed 30 % and escalate
   to whole-element — `dp6-eval-opus-5.md:55` already flagged this and it becomes
   *immediate* under this change, not hypothetical. Either raise the threshold or
   accept whole-element (which is what Porda effectively does anyway).

**Acceptance criterion.** On the fitness slice of the stratified set (§4.5): the
blurred region covers ≥ 70 % of the subject's visible torso bounding box on
≥ 90 % of positives, with the dev-split false-blur rate not worse than the M3
baseline. **Size: S** (settings default + ~40 lines in `region.ts` + tests).

**This is the first move.** It is the only change that directly addresses the
observed symptom ("faces blur, sports bras stay visible"), it uses code that
already exists and is already tested, and if M1 confirms the hypothesis it ships
in a day. Every other move below is weeks.

---

### 4.3 Move 2 — Invert the decision: person-first regions, NudeNet as escalation

**Competitor evidence.** Upstream's region mode gates *everything* on the person
detector and never consults NSFWJS when no person box exists (§2.4). Porda has
no secondary signal at all. In both, the part-level/explicitness signal
**escalates** but never **enables**.

**The change** — in `src/shared/policy.ts` and `src/content/region.ts`:

1. Region source becomes the **person instance** (YOLOv8n-seg mask, or the
   `coverageMode:"body"` ratio box when no mask), with gender attached via the
   existing `associateFacesToPersons` (`src/engine/person.ts:374`).
2. A person instance with an associated female face above `faceThreshold`
   (under `blurFemale`) blurs the **instance** — clothing state irrelevant, which
   is precisely the competitors' semantics.
3. NudeNet labels are demoted: `CORE_EXPOSED` escalates (to whole-element, or to
   a stricter cover style); `MODESTY_LABELS` become a *score input* on the
   instance, never a gate. `FEMALE_BREAST_COVERED`'s 0.7-on-everything
   miscalibration stops mattering because it stops being load-bearing.
4. Headless-torso case: an instance with **no** associated face falls back to
   NudeNet evidence on that instance (this is where `BELLY_EXPOSED` earns its
   keep) — the one case neither competitor handles well either.

**Acceptance criterion.** Per-type recall on the stratified set: fitness slice
recall ≥ 0.85 at overall precision ≥ 0.90; `FEMALE_BREAST_COVERED` removed from
every gating path; no regression on the art/portrait slices.
**Size: M** (policy rewrite + region source change + eval). Depends on M2.

---

### 4.4 Move 3 — Temporal tracking and evidence accumulation

**Competitor evidence.** Upstream's entire temporal machinery is `kt=1/He=2`
frame counters (`content.js:162`) plus an L2 pixel-diff skip
(`Lk.objectModelSkip`). Porda has none. **Both are stateless at the instance
level.** This is the clearest uncontested lead in the matrix.

**The change.** Mask-IoU / centroid greedy assignment across frames (~30 lines,
no model). Accumulate per-track evidence; decide on the track. Carry masks
forward with motion, so a missed pass translates the region instead of dropping
it. Replaces the current symmetric 2/2 hysteresis
(`src/content/video/sampler.ts:32`) with track-level voting, which permits
*lower* per-frame thresholds at equal precision.

**Acceptance criterion.** Blur-state transitions per minute on a steady scene
→ 0 (currently unmeasured); and a measured ability to drop the detector
threshold by ≥ 0.05 on video with no increase in track-level FP rate.
**Size: M.**

Steal alongside it, from upstream: the **L2 pixel-diff frame-skip** (cheap, and
strictly better than our scene-change-feeds-backoff) and Porda's **hold-down
timer** (hold full rate for N seconds after the last positive, not after the last
sample).

---

### 4.5 Move 4 — The stratified eval set

**Competitor evidence.** Neither competitor has an eval advantage to copy — Porda
publishes one number in a filename, upstream ships nothing. But §2 makes the
*content* of our set concrete: it must contain the distributions that make both
competitors look good and us look bad.

Target ~200 dev images, stratified and reported **per type**:
fitness/gym thumbnails (the flagship), headless/cropped torsos, back-to-camera,
swimwear, influencer selfie, group fitness (≥2 subjects — exercises the ≥2-faces
escalation), 2×2 collage thumbnails, text-overlay thumbnails, 9:16 crops,
anime/illustrated, **and a large clothed-women negative block** (the class whose
FP cost Move 1 and Move 2 raise). Plus a never-shrinking adversarial set seeded
by the current six fixtures.

Add the two product metrics neither competitor measures and neither review let
us off the hook for: **time-to-first-blur (p50/p95)** and **leak-duration per
page session**.

**Acceptance criterion.** The set exists, `scripts/eval.ts` reports per-slice
precision/recall, and every move above is scored on it.
**Size: M** (labour, not difficulty). **Must land before Move 2 ships.**

---

### 4.6 Move 5 — Zero-leak mode *(urgency raised by this study)*

Upstream ships `blurryStartMode` today (§2.9) with 20 s/6 s CSS animations,
default off. Porda has no concept of it (an OS overlay has no "before paint").
If upstream flips that default, the strongest differentiator both reviews
identified evaporates. Opt-in strict mode: in-viewport unverified media held
blurred/hidden until verdict; video starts blurred and clears after N verifying
frames. Composes with Move 3's tracking.
**Acceptance criterion:** measured leak-duration → 0 ms in strict mode on the
YouTube grid fixture. **Size: S–M.**

---

### 4.7 Moves 6+ — where we pass them rather than match them

Sequenced after parity, and only on evidence from §4.5:

- **Self-calibrated skin fraction inside the instance mask** — the FP control
  that lets us blur a sports bra and *not* blur a woman in a coat. Neither
  competitor can do this; both blur both.
- **CLIP/SigLIP arbitration in the uncertain band** — modesty as editable text.
  Structurally impossible for proprietary weights (§2.2's moat is the dataset).
- **Crop-classifier fine-tune on person crops** — upstream's own dormant `nF`
  path is the proof of design; frozen embeddings + logistic head, ~2–3 k crops.
  Our reveal telemetry (local-only) generates the labels for free.
- **Scholarly awrah presets as reviewable AGPL data files.**
- **Per-person hold-to-reveal**, which Move 3's instance identity unlocks.
- **Steal outright, cheap and uncontroversial:** benchmarked model-size selection
  (upstream's 60/100 ms tiers), dominant-background-colour cover (Porda), the
  CORS data-URL fallback with per-host give-up counter (upstream), and the
  hourly self-test inference (upstream).

### 4.8 Sequence

```
M1/M2/M3 measurement (1–2 d)
  └─▶ Move 1  region geometry parity        (S)   ── ships alone, immediately
        └─▶ Move 4  stratified eval set     (M)   ── gates everything below
              ├─▶ Move 2  policy inversion  (M)
              └─▶ Move 3  tracking          (M)   ── parallel-safe
                    └─▶ Move 5  zero-leak   (S–M)
                          └─▶ Moves 6+      (L)
```

### 4.9 The single highest-leverage first move

**Measure M1, then change the default `coverageMode` to `"body"`.**

Both competitors censor a whole figure derived from a whole-figure detection. We
censor a head-sized box derived from a face detection. On the exact content the
user reports as broken, that difference alone explains the observed behaviour —
faces blur, bodies don't — and the code to fix it is already written, already
tested (`src/content/region.ts:91-94`), and gated behind one default
(`src/shared/settings.ts:135`).

**Measure before touching it:** M1. If `FACE_FEMALE` fires at ≥ 0.4 on the
fitness fixtures and the head box covers a small fraction of the subject, this is
a configuration bug wearing a detection-quality costume, and everything in §4.3
onward is a *quality* programme rather than a *parity* programme. If
`FACE_FEMALE` does **not** fire on those fixtures, Move 1 is useless, the gap is
genuinely detection-level, and Move 2 jumps the queue.

Either way the answer costs an afternoon, and nothing else in this plan should be
funded before it exists.
