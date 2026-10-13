"use strict";
const screens = ["home", "capture", "result", "weather", "irrigation", "market"];
// Apache-2.0 Plant Disease model hosted by TensorFlow Hub (Rishit Dagli / Greenathon Plant AI).
// Model outputs logits for 38 PlantVillage classes; labels.json uses the corresponding class order.
const MODEL_URL = "https://tfhub.dev/rishit-dagli/tfjs-model/plant-disease/default/1";
const LABELS_URL = "./models/labels.json";
const IMAGE_SIZE = 224;
const MIN_CONFIDENCE = 0.60;
let selectedImage = null, model = null, labels = [], latestResult = null, loadPromise = null;

function showScreen(id) {
  screens.forEach(s => document.getElementById(s).classList.toggle("active", s === id));
  window.scrollTo(0, 0);
}
function setStatus(message) {
  const el = document.getElementById("analysisStatus");
  if (el) { el.textContent = message; el.classList.remove("hidden"); }
}
document.getElementById("photoInput").addEventListener("change", event => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  if (!file.type.startsWith("image/")) { setStatus("Please choose an image file."); return; }
  const preview = document.getElementById("preview");
  if (preview.dataset.objectUrl) URL.revokeObjectURL(preview.dataset.objectUrl);
  selectedImage = file;
  preview.dataset.objectUrl = URL.createObjectURL(file);
  preview.src = preview.dataset.objectUrl;
  preview.style.display = "block";
  document.getElementById("analyzeBtn").classList.remove("hidden");
  document.getElementById("analysisStatus").classList.add("hidden");
});
async function loadModel() {
  if (model && labels.length) return;
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    if (!window.tf) throw new Error("TensorFlow.js did not load. Check internet and refresh.");
    setStatus("Loading the trained model for the first time…");
    const [loadedModel, response] = await Promise.all([
      tf.loadLayersModel(MODEL_URL, { fromTFHub: true }),
      fetch(LABELS_URL, { cache: "no-store" })
    ]);
    if (!response.ok) { loadedModel.dispose(); throw new Error("Could not load models/labels.json."); }
    const loadedLabels = await response.json();
    const shape = loadedModel.outputs?.[0]?.shape;
    if (!Array.isArray(loadedLabels) || loadedLabels.length !== 38 || (shape && shape[shape.length - 1] !== loadedLabels.length)) {
      loadedModel.dispose(); throw new Error("Model output and class labels do not match.");
    }
    model = loadedModel; labels = loadedLabels;
  })();
  try { await loadPromise; } catch (e) { loadPromise = null; throw e; }
}
function parseLabel(label) {
  const parts = label.split("___");
  const crop = (parts[0] || "Unknown crop").replaceAll("_", " ").replaceAll(",", "").trim();
  const disease = (parts[1] || "Unknown class").replaceAll("_", " ").trim();
  return { crop, disease, healthy: /healthy/i.test(disease), raw: label };
}
function getGuidance(label) {
  const { crop, disease, healthy } = parseLabel(label);
  if (healthy) return {
    treatment: "The model classified this leaf as healthy. No disease treatment is indicated by this prediction. Continue normal crop care and confirm if symptoms appear.",
    prevention: "Inspect leaves regularly, keep tools clean, avoid unnecessary leaf wetness, and follow local agricultural guidance."
  };
  const virus = /virus|mosaic|yellow leaf curl/i.test(disease);
  const citrus = /huanglongbing|greening/i.test(disease);
  const mite = /spider mites/i.test(disease);
  const mildew = /mildew|powdery/i.test(disease);
  let treatment = `The model suggests ${disease} on ${crop}. Remove badly affected leaves only when appropriate, avoid moving infected plant material, and ask a local agricultural extension officer to confirm before applying any treatment.`;
  let prevention = "Keep tools clean, avoid overhead watering where possible, provide suitable airflow, monitor nearby plants, and rotate crops where appropriate.";
  if (virus) treatment = `The model suggests a virus-related condition (${disease}) on ${crop}. Viral diseases often have no direct cure. Isolate suspicious plants where practical and contact a local agricultural officer for confirmation and vector-management advice; do not apply pesticides without local guidance.`;
  else if (citrus) treatment = "The model suggests citrus greening (Huanglongbing). There is no simple curative treatment; contact your local agriculture or horticulture department promptly for confirmation and locally recommended management.";
  else if (mite) treatment = "The model suggests spider-mite damage. Check the underside of leaves and ask a local agricultural officer to confirm the cause and advise on an appropriate locally approved control method.";
  else if (mildew) prevention = "Improve airflow, avoid crowding plants, inspect new growth, and water near the soil rather than wetting leaves when possible.";
  return { treatment, prevention };
}
async function analyze() {
  if (!selectedImage) { setStatus("முதலில் ஒரு இலைப் புகைப்படத்தைத் தேர்வு செய்யவும்."); return; }
  const button = document.getElementById("analyzeBtn"); button.disabled = true;
  let bitmap = null, prediction = null, probabilities = null;
  try {
    await loadModel(); setStatus("Analyzing the leaf photo…");
    bitmap = await createImageBitmap(selectedImage);
    probabilities = tf.tidy(() => {
      const input = tf.browser.fromPixels(bitmap).resizeBilinear([IMAGE_SIZE, IMAGE_SIZE]).toFloat().div(255).expandDims(0);
      const raw = model.predict(input);
      const logits = Array.isArray(raw) ? raw[0] : raw;
      return tf.softmax(logits).squeeze();
    });
    const scores = Array.from(await probabilities.data());
    if (scores.length !== labels.length || scores.some(v => !Number.isFinite(v))) throw new Error("Invalid model output or label mismatch.");
    let best = 0; for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
    latestResult = { label: labels[best], confidence: scores[best] };
    renderResult(latestResult.label, latestResult.confidence);
    document.getElementById("analysisStatus").classList.add("hidden"); showScreen("result");
  } catch (error) {
    console.error(error);
    setStatus("AI prediction unavailable: " + (error?.message || "unknown error") + ". Check internet access and try again.");
  } finally {
    if (bitmap?.close) bitmap.close(); if (prediction) prediction.dispose(); if (probabilities) probabilities.dispose(); button.disabled = false;
  }
}
function renderResult(label, confidence) {
  const wrap = document.getElementById("resultImageWrap"); wrap.replaceChildren();
  const img = document.createElement("img"); img.alt = "Analyzed leaf"; img.style.cssText = "width:100%;max-height:220px;object-fit:cover;border-radius:12px";
  img.src = URL.createObjectURL(selectedImage); wrap.appendChild(img);
  const parsed = parseLabel(label), low = confidence < MIN_CONFIDENCE, guidance = getGuidance(label);
  document.getElementById("predictedCrop").textContent = parsed.crop;
  document.getElementById("predictedClass").textContent = low ? "Uncertain — try a clearer photo" : parsed.disease;
  document.getElementById("confidence").textContent = `${(confidence * 100).toFixed(1)}%`;
  document.getElementById("resultTag").textContent = low ? "Low confidence" : "Model prediction";
  document.getElementById("resultNote").textContent = low
    ? "The model confidence is low. No disease-specific treatment should be chosen from this result. Try a clear photo of one leaf in natural light."
    : "AI screening only — not a confirmed diagnosis. The model was trained on PlantVillage-style images and may be less reliable on field photos or crops outside its 38 classes.";
  document.getElementById("treatmentText").textContent = low ? "No disease-specific treatment is suggested because the prediction is uncertain. Ask a local agricultural officer." : guidance.treatment;
  document.getElementById("preventionText").textContent = guidance.prevention;
}
function startVoice() {
  const status = document.getElementById("voiceStatus");
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) { status.textContent = "இந்த browser-ல் voice input ஆதரவு இல்லை. Chrome-ல் முயற்சி செய்யவும்."; return; }
  const recognizer = new SpeechRecognition(); recognizer.lang = "ta-IN"; recognizer.interimResults = false;
  status.textContent = "🎤 கேட்கிறேன்…";
  recognizer.onresult = e => status.textContent = "நீங்கள் சொன்னது: " + e.results[0][0].transcript;
  recognizer.onerror = () => status.textContent = "Voice input தொடங்க முடியவில்லை. Microphone permission-ஐ சரிபார்க்கவும்.";
  recognizer.onend = () => { if (status.textContent === "🎤 கேட்கிறேன்…") status.textContent = "Voice input முடிந்தது."; };
  try { recognizer.start(); } catch (_) { status.textContent = "Voice input மீண்டும் தொடங்க முடியவில்லை."; }
}
function speakTamil() {
  if (!("speechSynthesis" in window)) { alert("இந்த browser-ல் Tamil voice output ஆதரவு இல்லை."); return; }
  if (!latestResult) return;
  const p = parseLabel(latestResult.label);
  const msg = latestResult.confidence < MIN_CONFIDENCE
    ? "இந்த கணிப்பு உறுதியானதல்ல. தெளிவான புகைப்படம் எடுத்து மீண்டும் முயற்சி செய்யவும். வேளாண் அதிகாரியிடம் ஆலோசனை பெறவும்."
    : `பயிர்: ${p.crop}. கணிப்பு: ${p.disease}. நம்பிக்கை அளவு ${(latestResult.confidence * 100).toFixed(0)} சதவீதம். இது உறுதியான நோய் கண்டறிதல் அல்ல. சிகிச்சைக்கு முன் வேளாண் நிபுணரை அணுகவும்.`;
  const utterance = new SpeechSynthesisUtterance(msg); utterance.lang = "ta-IN"; speechSynthesis.cancel(); speechSynthesis.speak(utterance);
}
const MODEL_URL = "https://raw.githubusercontent.com/Rishit-dagli/Greenathon-Plant-AI/main/models/plant_disease_tfjs/model.json";
const LABELS_URL = "https://raw.githubusercontent.com/Rishit-dagli/Greenathon-Plant-AI/main/models/plant_disease_tfjs/class_indices.json";
const labelMap = await response.json();
const loadedLabels = Array.isArray(labelMap)
  ? labelMap
  : Object.keys(labelMap)
      .sort((a, b) => Number(a) - Number(b))
      .map(key => labelMap[key]);
