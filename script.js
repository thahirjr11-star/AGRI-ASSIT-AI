"use strict";

const MODEL_URL =
  "https://raw.githubusercontent.com/Rishit-dagli/Greenathon-Plant-AI/main/models/plant_disease_tfjs/model.json";

const LABELS_URL =
  "https://raw.githubusercontent.com/Rishit-dagli/Greenathon-Plant-AI/main/models/plant_disease_tfjs/class_indices.json";

const IMAGE_SIZE = 224;
const MIN_CONFIDENCE = 0.60;

let selectedImage = null;
let model = null;
let labels = [];
let latestResult = null;
let loadPromise = null;

const screens = [
  "home", "capture", "result",
  "weather", "irrigation", "market"
];

function showScreen(id) {
  screens.forEach(screen => {
    const element = document.getElementById(screen);
    if (element) {
      element.classList.toggle("active", screen === id);
    }
  });

  window.scrollTo(0, 0);
}

function setStatus(message) {
  const element = document.getElementById("analysisStatus");

  if (element) {
    element.textContent = message;
    element.classList.remove("hidden");
  }
}

const photoInput = document.getElementById("photoInput");

if (photoInput) {
  photoInput.addEventListener("change", event => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setStatus("Please select a valid image.");
      return;
    }

    const preview = document.getElementById("preview");

    if (preview.dataset.objectUrl) {
      URL.revokeObjectURL(preview.dataset.objectUrl);
    }

    selectedImage = file;
    preview.dataset.objectUrl = URL.createObjectURL(file);
    preview.src = preview.dataset.objectUrl;
    preview.style.display = "block";

    document.getElementById("analyzeBtn")
      ?.classList.remove("hidden");

    document.getElementById("analysisStatus")
      ?.classList.add("hidden");
  });
}

async function loadModel() {
  if (model && labels.length === 38) return;
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    if (!window.tf) {
      throw new Error("TensorFlow.js did not load.");
    }

    setStatus("Loading AI model...");

    const [loadedModel, response] = await Promise.all([
      tf.loadLayersModel(MODEL_URL),
      fetch(LABELS_URL)
    ]);

    if (!response.ok) {
      loadedModel.dispose();
      throw new Error("Could not load disease labels.");
    }

    const labelMap = await response.json();

    const loadedLabels = Array.isArray(labelMap)
      ? labelMap
      : Object.keys(labelMap)
          .sort((a, b) => Number(a) - Number(b))
          .map(key => labelMap[key]);

    const shape = loadedModel.outputs?.[0]?.shape;
    const count = shape ? shape[shape.length - 1] : null;

    if (loadedLabels.length !== 38 || count !== 38) {
      loadedModel.dispose();
      throw new Error("Model and labels do not match.");
    }

    model = loadedModel;
    labels = loadedLabels;
  })();

  try {
    await loadPromise;
  } catch (error) {
    loadPromise = null;
    throw error;
  }
}

function parseLabel(label) {
  const parts = label.split("___");

  return {
    crop: (parts[0] || "Unknown crop")
      .replaceAll("_", " ")
      .trim(),

    disease: (parts[1] || "Unknown condition")
      .replaceAll("_", " ")
      .trim(),

    healthy: /healthy/i.test(parts[1] || "")
  };
}

function getGuidance(label) {
  const { crop, disease, healthy } = parseLabel(label);

  if (healthy) {
    return {
      treatment:
        "The model classified this leaf as healthy. " +
        "Continue normal crop care and monitor regularly.",

      prevention:
        "Inspect leaves regularly, keep tools clean, " +
        "and follow local agricultural guidance."
    };
  }

  return {
    treatment:
      `The model suggests ${disease} on ${crop}. ` +
      "Ask a local agricultural officer to confirm " +
      "the diagnosis before choosing a treatment.",

    prevention:
      "Keep tools clean, monitor nearby plants, " +
      "provide suitable airflow, and avoid unnecessary " +
      "leaf wetness."
  };
}

async function analyze() {
  if (!selectedImage) {
    setStatus("முதலில் ஒரு இலைப் புகைப்படத்தைத் தேர்வு செய்யவும்.");
    return;
  }

  const button = document.getElementById("analyzeBtn");
  if (button) button.disabled = true;

  let bitmap = null;
  let prediction = null;

  try {
    await loadModel();

    setStatus("Analyzing leaf photo...");

    bitmap = await createImageBitmap(selectedImage);

    prediction = tf.tidy(() => {
      const input = tf.browser
        .fromPixels(bitmap)
        .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])
        .toFloat()
        // MobileNet-style input normalization
        .div(127.5)
        .sub(1)
        .expandDims(0);

      const output = model.predict(input);
      return Array.isArray(output) ? output[0] : output;
    });

    const scores = Array.from(await prediction.data());

    if (
      scores.length !== labels.length ||
      scores.some(value => !Number.isFinite(value))
    ) {
      throw new Error("Invalid prediction values.");
    }

    let bestIndex = 0;

    for (let i = 1; i < scores.length; i++) {
      if (scores[i] > scores[bestIndex]) {
        bestIndex = i;
      }
    }

    const rawScore = scores[bestIndex];

    // This model is expected to output class probabilities.
    // If the output is not probability-like, convert it to
    // probabilities rather than applying softmax twice.
    let confidence;

    const valuesLookLikeProbabilities =
      scores.every(value => value >= 0 && value <= 1) &&
      Math.abs(scores.reduce((sum, value) => sum + value, 0) - 1) < 0.05;

    if (valuesLookLikeProbabilities) {
      confidence = rawScore;
    } else {
      const maxValue = Math.max(...scores);
      const expValues = scores.map(value =>
        Math.exp(value - maxValue)
      );
      const total = expValues.reduce((sum, value) => sum + value, 0);

      confidence = total > 0 ? expValues[bestIndex] / total : 0;
    }

    if (confidence <= 0 || !Number.isFinite(confidence)) {
      throw new Error(
        "The model returned no usable confidence. Please try another photo."
      );
    }

    latestResult = {
      label: labels[bestIndex],
      confidence
    };

    renderResult(latestResult.label, confidence);

    document.getElementById("analysisStatus")
      ?.classList.add("hidden");

    showScreen("result");

  } catch (error) {
    console.error("AI prediction error:", error);

    setStatus(
      "AI prediction unavailable: " +
      (error.message || "Unknown error") +
      ". Please refresh and try again."
    );

  } finally {
    if (bitmap) bitmap.close();
    if (prediction) prediction.dispose();
    if (button) button.disabled = false;
  }
}

function renderResult(label, confidence) {
  const wrap = document.getElementById("resultImageWrap");

  if (wrap) {
    wrap.replaceChildren();

    const image = document.createElement("img");
    image.alt = "Analyzed leaf";
    image.style.cssText =
      "width:100%;max-height:220px;" +
      "object-fit:cover;border-radius:12px";

    image.src = URL.createObjectURL(selectedImage);
    wrap.appendChild(image);
  }

  const parsed = parseLabel(label);
  const uncertain = confidence < MIN_CONFIDENCE;
  const guidance = getGuidance(label);

  document.getElementById("predictedCrop").textContent =
    parsed.crop;

  document.getElementById("predictedClass").textContent =
    uncertain
      ? "Uncertain - try a clearer photo"
      : parsed.disease;

  document.getElementById("confidence").textContent =
    (confidence * 100).toFixed(1) + "%";

  document.getElementById("resultTag").textContent =
    uncertain ? "Low confidence" : "Model prediction";

  document.getElementById("resultNote").textContent =
    uncertain
      ? "The prediction is uncertain. Try one clear leaf " +
        "in natural light. Results are not a confirmed diagnosis."
      : "AI screening only, not a confirmed diagnosis. " +
        "Consult an agricultural expert before treatment.";

  document.getElementById("treatmentText").textContent =
    uncertain
      ? "No disease-specific treatment is suggested. " +
        "Ask a local agricultural officer."
      : guidance.treatment;

  document.getElementById("preventionText").textContent =
    guidance.prevention;
}

function startVoice() {
  const status = document.getElementById("voiceStatus");

  const SpeechRecognition =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    status.textContent =
      "இந்த browser-ல் voice input இல்லை. Chrome-ல் முயற்சி செய்யவும்.";
    return;
  }

  const recognizer = new SpeechRecognition();
  recognizer.lang = "ta-IN";
  recognizer.interimResults = false;

  status.textContent = "🎤 கேட்கிறேன்...";

  recognizer.onresult = event => {
    status.textContent =
      "நீங்கள் சொன்னது: " +
      event.results[0][0].transcript;
  };

  recognizer.onerror = () => {
    status.textContent =
      "Voice input failed. Check microphone permission.";
  };

  recognizer.onend = () => {
    if (status.textContent === "🎤 கேட்கிறேன்...") {
      status.textContent = "Voice input முடிந்தது.";
    }
  };

  try {
    recognizer.start();
  } catch (error) {
    status.textContent = "Voice input மீண்டும் தொடங்க முடியவில்லை.";
  }
}

function speakTamil() {
  if (!("speechSynthesis" in window)) {
    alert("Tamil voice output is not supported in this browser.");
    return;
  }

  if (!latestResult) return;

  const parsed = parseLabel(latestResult.label);

  const message =
    latestResult.confidence < MIN_CONFIDENCE
      ? "இந்த கணிப்பு உறுதியானதல்ல. " +
        "தெளிவான புகைப்படம் எடுத்து மீண்டும் முயற்சி செய்யவும். " +
        "வேளாண் அதிகாரியிடம் ஆலோசனை பெறவும்."
      : `பயிர்: ${parsed.crop}. ` +
        `கணிப்பு: ${parsed.disease}. ` +
        `நம்பிக்கை அளவு ${(latestResult.confidence * 100).toFixed(0)} சதவீதம். ` +
        "இது உறுதியான நோய் கண்டறிதல் அல்ல. " +
        "வேளாண் நிபுணரை அணுகவும்.";

  const utterance = new SpeechSynthesisUtterance(message);
  utterance.lang = "ta-IN";

  speechSynthesis.cancel();
  speechSynthesis.speak(utterance);
}
