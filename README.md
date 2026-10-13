# Agri Assist AI — Leaf Disease Screening

A mobile-friendly static website for leaf-photo screening using a real pretrained TensorFlow.js model hosted on TensorFlow Hub.

## AI model and labels
- Model: Rishit Dagli / Greenathon Plant AI Plant Disease model, served from TensorFlow Hub.
- Model card and model source: https://github.com/Rishit-dagli/Greenathon-Plant-AI
- Published class-index reference: https://github.com/Rishit-dagli/Greenathon-Plant-AI/releases/download/v0.1.0/class_indices.json
- Model format: TensorFlow.js LayersModel loaded from TensorFlow Hub.
- Input: RGB image resized to 224 × 224 and scaled to [0, 1].
- Output: 38 logits; the app applies softmax and maps the winning index to `models/labels.json`.
- The upstream model is published under Apache 2.0 and documents the standard 38 PlantVillage classes. The label file is included locally so the website does not have to fetch labels at runtime. Before a production release, compare it against the upstream class-index file above and run a held-out image test.
- Model weights are hosted by TensorFlow Hub and are not copied into this ZIP. First prediction requires internet access.

## Run
1. Extract the ZIP.
2. Open the folder in a terminal and run `python -m http.server 8000`.
3. Visit `http://localhost:8000`.
4. Allow internet access and choose a clear leaf image with one leaf in focus.
5. Press **Analyze with AI**.

Opening `index.html` directly as a `file://` URL may block model/label loading. Use a local HTTP server or GitHub Pages.

## Safety and limitations
- This is a screening aid, not a confirmed diagnosis or substitute for a local agricultural officer.
- The pretrained model covers 38 PlantVillage categories across 14 crop types; it does not cover every crop or disease in Tamil Nadu.
- PlantVillage images often have cleaner backgrounds than real field photos, so published benchmark accuracy must not be treated as guaranteed field accuracy.
- Low-confidence predictions suppress disease-specific treatment guidance. For confident predictions, guidance is conservative and recommends local confirmation before treatment.
- Weather, irrigation and market-price cards remain clearly labelled demo features; live APIs are not connected.
- Tamil speech recognition and speech synthesis depend on browser/device support and microphone permission.

## Testing performed in this package
- JavaScript syntax check with Node.js.
- Class-label count and index uniqueness checks.
- Static consistency check: model output count expected to be 38 and label list contains 38 entries.

## Still required before calling this production-verified
Run this site in a real browser with internet access; test a set of separately labelled healthy and diseased images from multiple supported crops; record a confusion matrix/accuracy on held-out field-like photos; test Tamil speech on the target phone; and verify the TensorFlow Hub model and label ordering against the official release mapping. Do not claim field accuracy without these tests.
