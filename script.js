
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

  let crop = (parts[0] || "Unknown crop")
    .replaceAll("_", " ")
    .trim();

  const disease = (parts[1] || "Unknown condition")
    .replaceAll("_", " ")
    .trim();

  // Display Apple as Mango.
  // Note: this changes the displayed name only.
  if (/^apple$/i.test(crop)) {
    crop = "Mango";
  }

  return {
    crop,
    disease,
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
      "Confirm the diagnosis with a local agricultural officer " +
      "before choosing treatment.",

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

    const isProbabilityOutput =
      scores.every(value => value >= 0 && value <= 1) &&
      Math.abs(
        scores.reduce((sum, value) => sum + value, 0) - 1
      ) < 0.05;

    let confidence;

    if (isProbabilityOutput) {
      confidence = rawScore;
    } else {
      const maxValue = Math.max(...scores);

      const expValues = scores.map(value =>
        Math.exp(value - maxValue)
      );

      const total = expValues.reduce(
        (sum, value) => sum + value, 0
      );

      confidence = total > 0
        ? expValues[bestIndex] / total
        : 0;
    }

    if (confidence <= 0 || !Number.isFinite(confidence)) {
      throw new Error("Invalid model confidence.");
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

  if (wrap && selectedImage) {
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
    parsed.disease;

  document.getElementById("confidence").textContent =
    (confidence * 100).toFixed(1) + "%";

  document.getElementById("resultTag").textContent =
    uncertain ? "Low Confidence" : "AI Prediction";

  document.getElementById("resultNote").textContent =
    uncertain
      ? "Possible result: " + parsed.disease +
        ". Confidence is low, so this may be incorrect. " +
        "Please confirm with an agricultural expert."
      : "AI screening result: " + parsed.disease +
        ". This is not a confirmed diagnosis.";

  document.getElementById("treatmentText").textContent =
    uncertain
      ? "Disease not confirmed. Do not select treatment based only on this prediction. Consult a local agricultural officer."
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
    status.textContent =
      "Voice input மீண்டும் தொடங்க முடியவில்லை.";
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
      ? `இந்த இலைக்கான கணிப்பு ${parsed.disease}. ` +
        "நம்பிக்கை அளவு குறைவாக உள்ளது. " +
        "இது தவறாக இருக்கலாம். " +
        "வேளாண் நிபுணரிடம் உறுதிப்படுத்தவும்."
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
// 🌦️ WEATHER ALERT
let weatherTamil = "";

async function getWeather() {
  const city = document.getElementById("city").value.trim();
  const result = document.getElementById("weatherResult");

  if (!city) {
    result.innerHTML = "Please enter a city name.";
    return;
  }

  result.innerHTML = "Loading weather...";

  try {
    const geoURL =
      `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1&language=en&format=json`;

    const geoResponse = await fetch(geoURL);
    const geoData = await geoResponse.json();

    if (!geoData.results || geoData.results.length === 0) {
      result.innerHTML = "City not found. Try another city.";
      return;
    }

    const place = geoData.results[0];

    const weatherURL =
      `https://api.open-meteo.com/v1/forecast?latitude=${place.latitude}&longitude=${place.longitude}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m`;

    const weatherResponse = await fetch(weatherURL);
    const weatherData = await weatherResponse.json();

    const w = weatherData.current;

    const descriptions = {
      0: "Clear sky",
      1: "Mainly clear",
      2: "Partly cloudy",
      3: "Cloudy",
      45: "Foggy",
      48: "Foggy",
      51: "Light drizzle",
      53: "Drizzle",
      55: "Heavy drizzle",
      61: "Light rain",
      63: "Rain",
      65: "Heavy rain",
      80: "Rain showers",
      81: "Rain showers",
      82: "Heavy rain showers",
      95: "Thunderstorm"
    };

    const condition =
      descriptions[w.weather_code] || "Variable weather";

    result.innerHTML = `
      <h3>📍 ${place.name}, ${place.country}</h3>
      <p>🌡️ Temperature: ${w.temperature_2m}°C</p>
      <p>🤒 Feels Like: ${w.apparent_temperature}°C</p>
      <p>💧 Humidity: ${w.relative_humidity_2m}%</p>
      <p>🌧️ Rainfall: ${w.precipitation} mm</p>
      <p>🌤️ Condition: ${condition}</p>
      <p>💨 Wind Speed: ${w.wind_speed_10m} km/h</p>
    `;

    weatherTamil =
      `${place.name}. வெப்பநிலை ${w.temperature_2m} டிகிரி செல்சியஸ். ` +
      `காற்றின் ஈரப்பதம் ${w.relative_humidity_2m} சதவீதம். ` +
      `மழையளவு ${w.precipitation} மில்லிமீட்டர். ` +
      `வானிலை ${condition}.`;

  } catch (error) {
    result.innerHTML =
      "Weather unavailable. Check your internet connection and try again.";
  }
}


// 🔊 TAMIL VOICE
function speakTamil(text) {
  if (!("speechSynthesis" in window)) {
    alert("Tamil voice is not supported in this browser.");
    return;
  }

  window.speechSynthesis.cancel();

  const message = new SpeechSynthesisUtterance(text);
  message.lang = "ta-IN";
  message.rate = 0.85;

  const voices = window.speechSynthesis.getVoices();

  const tamilVoice = voices.find(voice =>
    voice.lang.toLowerCase().startsWith("ta")
  );

  if (tamilVoice) {
    message.voice = tamilVoice;
  }

  window.speechSynthesis.speak(message);
}

function speakWeather() {
  if (!weatherTamil) {
    speakTamil("முதலில் வானிலை விவரங்களைப் பெறவும்.");
    return;
  }

  speakTamil(weatherTamil);
}


// 💧 IRRIGATION GUIDANCE
let irrigationTamil = "";

function getIrrigation() {
  const crop = document.getElementById("crop").value;
  const result = document.getElementById("irrigationResult");

  const advice = {
    rice: {
      title: "Rice / நெல்",
      text: "Check field water level regularly. Avoid unnecessary continuous flooding.",
      tamil: "நெல் வயலில் நீர்மட்டத்தைத் தொடர்ந்து கண்காணிக்கவும். தேவையற்ற நீர் தேக்கத்தைத் தவிர்க்கவும்."
    },

    tomato: {
      title: "Tomato / தக்காளி",
      text: "Keep soil moisture consistent. Avoid waterlogging near the roots.",
      tamil: "தக்காளிச் செடிகளுக்கு மண்ணில் சீரான ஈரப்பதத்தைப் பராமரிக்கவும். வேர்களில் நீர் தேங்குவதைத் தவிர்க்கவும்."
    },

    mango: {
      title: "Mango / மாம்பழம்",
      text: "Water according to soil dryness, tree age and rainfall.",
      tamil: "மண்ணின் வறட்சி, மரத்தின் வயது மற்றும் மழையைப் பொறுத்து நீர் பாய்ச்சவும்."
    },

    banana: {
      title: "Banana / வாழை",
      text: "Check soil moisture frequently, especially during hot weather.",
      tamil: "குறிப்பாக வெயில் காலத்தில் வாழைச் செடிகளின் மண்ணில் ஈரப்பதம் உள்ளதா என்று அடிக்கடி பார்க்கவும்."
    },

    groundnut: {
      title: "Groundnut / நிலக்கடலை",
      text: "Avoid both water shortage and waterlogging. Monitor soil moisture during flowering and pod development.",
      tamil: "நிலக்கடலைப் பயிரில் நீர்ப்பற்றாக்குறையையும் நீர் தேக்கத்தையும் தவிர்க்கவும். பூக்கும் மற்றும் காய் உருவாகும் காலத்தில் மண்ணின் ஈரப்பதத்தைக் கண்காணிக்கவும்."
    },

    cotton: {
      title: "Cotton / பருத்தி",
      text: "Water according to soil moisture and crop growth stage. Ensure proper drainage.",
      tamil: "பருத்திப் பயிரின் வளர்ச்சி நிலை மற்றும் மண்ணின் ஈரப்பதத்திற்கு ஏற்ப நீர் பாய்ச்சவும். நீர் வடிகால் வசதியை உறுதி செய்யவும்."
    }
  };

  const selected = advice[crop];

  result.innerHTML = `
    <h3>🌱 ${selected.title}</h3>
    <p>${selected.text}</p>
    <small>General guidance only. Check local soil and weather conditions.</small>
  `;

  irrigationTamil = selected.tamil;
}

function speakIrrigation() {
  if (!irrigationTamil) {
    speakTamil("முதலில் பயிரைத் தேர்ந்தெடுத்து வழிகாட்டுதலைப் பெறவும்.");
    return;
  }

  speakTamil(irrigationTamil);
}
