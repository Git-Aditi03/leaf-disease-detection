"""
Leaf Disease Detection — HuggingFace Space
==========================================
Includes:
- Suppressed TF logs
- EfficientNet preprocessing
- Matplotlib Confidence Bar Chart
- Severity Score Badge & Confidence Warning
- Hindi Translation (deep-translator)
- Voice Output (gTTS)
- Downloadable Text Report
- Integrated Live Visual Crossing Weather API
"""

import os
# ── Silence TensorFlow log noise BEFORE importing tensorflow ────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import numpy as np
import cv2
import requests
import tempfile
import datetime
import random
from PIL import Image
import tensorflow as tf
from transformers import pipeline
import matplotlib.pyplot as plt
import gradio as gr

# Import our custom UI layout
from ui_components import create_comprehensive_ui

# ── Stabilize Model Accuracy (Fixed Seeds) ────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ── Optional Dependencies ─────────────────────────────────────────────────
try:
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source='auto', target='hi')
except ImportError:
    translator = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

# ══════════════════════════════════════════════════════════════════
# 1. Load model & data files
# ══════════════════════════════════════════════════════════════════
print("Loading disease classification model ...")
model = tf.keras.models.load_model("leaf_disease_model.keras")

print("Loading Leaf Gatekeeper (CLIP zero-shot) ...")
leaf_checker = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
)

def _load_json(preferred, fallback):
    for path in (preferred, fallback):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    raise FileNotFoundError(f"Could not find '{preferred}' or '{fallback}'.")

class_names = _load_json("class_names.json", "class_names (2).json")
disease_info = _load_json("disease_info.json", "disease_info (1).json")

# ══════════════════════════════════════════════════════════════════
# 2. Helper utilities
# ══════════════════════════════════════════════════════════════════

def _class_name(idx):
    return class_names[idx] if isinstance(class_names, list) else class_names[str(idx)]

def preprocess_image(image):
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image, dtype="float32")
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def generate_visual_heatmap(pil_image):
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    heatmap = cv2.applyColorMap(dilated, cv2.COLORMAP_JET)
    result = cv2.addWeighted(cv_img, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

def get_real_weather_and_risk(location):
    API_KEY = "BFEMY972NDGM8SBRML49PGGPF"
    location = location.strip() or "Purnia, Bihar"
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/"
        f"timeline/{location}?unitGroup=metric&key={API_KEY}&contentType=json"
    )
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        today = resp.json()["days"][0]
        return {
            "temp": today["temp"],
            "humidity": today["humidity"],
            "conditions": today["conditions"],
            "rain": today["precip"] > 0,
            "location": location,
        }
    except Exception:
        return None

def format_weather_markdown(weather, location):
    if weather is None:
        return f"⚠️ Weather unavailable for **{location}**."
    risk = "🔴 High Risk" if weather["humidity"] > 75 else "🟢 Low Risk"
    rain_str = " | 🌧️ Rain" if weather["rain"] else ""
    return (
        f"🌡️ **Weather in {weather['location']}:** {weather['temp']}°C, "
        f"{weather['conditions']}{rain_str} \n"
        f"💧 **Humidity:** {weather['humidity']}% \n"
        f"🍄 **Spread Risk:** {risk}"
    )

def get_spray_timing(disease_key, weather, base_days):
    if weather is None:
        return f"Spray within {base_days} days."
    h, rain = weather["humidity"], weather["rain"]
    if any(k in disease_key for k in ("Late_blight", "mosaic_virus")):
        return "⚠️ **URGENT:** Act within 1–2 days!" if rain or h > 85 else f"Act within {base_days} days."
    if rain and h > 80:
        return f"Spray within {max(1, base_days - 4)} days (Rain + Humidity)."
    return f"Spray within {base_days} days."

def format_disease_details(detected_diseases, weather):
    parts = []
    for entry in detected_diseases:
        name, conf = entry["name"], entry["confidence"]
        info = disease_info.get(name, disease_info.get("default", {}))
        is_healthy = "healthy" in name.lower()
        emoji = "✅" if is_healthy else "🦠"
        timing = get_spray_timing(name, weather, info.get("base_days", 10))

        block = f"#### {emoji} {info.get('simple_name', name)} `{conf:.1%}`\n\n"
        if not is_healthy:
            block += (
                f"**Cause:** {info.get('cause', '—')} | "
                f"**Region:** {info.get('region', '—')} \n"
                f"**⏰ Spray timing:** {timing}\n\n"
            )

        pesticides = info.get("pesticide", [])
        # pesticides must be a list of dicts
        if pesticides and isinstance(pesticides, list):
            block += "**🧪 Pesticide Recommendations**\n| Product | Dose | When |\n|---|---|---|\n"
            for p in pesticides:
                if isinstance(p, dict):
                    block += f"| {p.get('product','—')} | {p.get('dose','—')} | {p.get('when','—')} |\n"

        organic = info.get("organic_option")
        if organic:
            block += f"\n🌱 **Organic option:** {organic}\n"
        parts.append(block)
    return "\n---\n\n".join(parts)

# ══════════════════════════════════════════════════════════════════
# 3. Enhanced Tool Features
# ══════════════════════════════════════════════════════════════════

def plot_confidence(detected_list):
    fig, ax = plt.subplots(figsize=(6, 3))
    names = [d["name"].replace("_", " ") for d in detected_list][::-1]
    confs = [d["confidence"] * 100 for d in detected_list][::-1]

    colors = ['#ff9999' if c < 50 else '#66b3ff' for c in confs]
    if names and 'healthy' in names[-1].lower():
        colors[-1] = '#99ff99'

    ax.barh(names, confs, color=colors)
    ax.set_xlabel('Confidence (%)')
    ax.set_xlim(0, 100)
    ax.set_title('Prediction Confidence Levels')
    plt.tight_layout()
    return fig

def generate_report(status_md, details_md):
    content = (
        f"LEAF DISEASE DIAGNOSIS REPORT\n"
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    )
    content += status_md.replace('*', '').replace('#', '') + "\n\n"
    content += "TREATMENT PLAN:\n" + details_md.replace('*', '').replace('#', '') + "\n"

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    with open(temp_file.name, "w", encoding="utf-8") as f:
        f.write(content)
    return temp_file.name

def translate_to_hindi(text):
    if not translator:
        return "⚠️ Please add `deep-translator` to requirements.txt"
    try:
        translated = translator.translate(text[:800])
        return translated + "..." if len(text) > 800 else translated
    except Exception as e:
        return f"⚠️ Translation failed: {str(e)}"

def generate_voice(prediction_name, confidence):
    if not gTTS:
        return None
    try:
        text = (
            f"The system has detected {prediction_name.replace('_', ' ')} "
            f"with {confidence*100:.0f} percent confidence."
        )
        tts = gTTS(text, lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════
# 4. Main prediction function
# ══════════════════════════════════════════════════════════════════

def predict_fn(image, location, gradcam_check):
    if image is None:
        return None, "⚠️ Upload an image.", "", "", None, "", None, "", None

    gk_results = leaf_checker(
        image,
        candidate_labels=["a close up of a plant leaf", "food", "person", "background"]
    )
    if gk_results[0]["label"] != "a close up of a plant leaf":
        return (
            image,
            "🚫 **Not a Leaf**\n\nPlease upload a clear image of a plant leaf to get accurate results.",
            "", "", None, "", None, "", None
        )

    weather = get_real_weather_and_risk(location)
    weather_md = format_weather_markdown(weather, location or "Purnia, Bihar")

    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)[0]

    detected = [
        {"name": _class_name(idx), "confidence": float(conf)}
        for idx, conf in enumerate(predictions)
        if conf >= 0.25
    ]
    detected.sort(key=lambda x: x["confidence"], reverse=True)
    if not detected:
        fb_idx = int(np.argmax(predictions))
        detected = [{"name": _class_name(fb_idx), "confidence": float(predictions[fb_idx])}]
    detected = detected[:3]

    top = detected[0]
    is_healthy = "healthy" in top["name"].lower()

    # ── Confidence Warning ──
    warning = ""
    if top['confidence'] < 0.60:
        warning = "\n\n⚠️ **Low Confidence:** The AI is unsure. Try a clearer image with better lighting."

    # ── Severity Badge ──
    base_sev = 0 if is_healthy else top["confidence"] * 100
    if weather and weather.get("humidity", 0) > 80 and not is_healthy:
        base_sev += 15
    if base_sev == 0:
        sev_badge = "🟢 **Severity:** None (Healthy)"
    elif base_sev < 50:
        sev_badge = "🟡 **Severity:** Mild"
    elif base_sev < 80:
        sev_badge = "🟠 **Severity:** Moderate"
    else:
        sev_badge = "🔴 **Severity:** Critical"

    output_image = generate_visual_heatmap(image) if gradcam_check else image

    if is_healthy:
        status_md = f"### ✅ Healthy Leaf\n**{top['name']}** ({top['confidence']:.1%})"
    else:
        status_md = "### ⚠️ Detected Conditions:\n" + "\n".join(
            f"- **{d['name']}** ({d['confidence']:.1%})" for d in detected
        )

    details_md = format_disease_details(detected, weather)
    conf_plot = plot_confidence(detected)
    audio_path = generate_voice(top['name'], top['confidence'])
    hindi_txt = translate_to_hindi(
        f"मुख्य खोज (Top Result): {top['name'].replace('_', ' ')} ({top['confidence']:.1%})"
    )
    report_file = generate_report(status_md, details_md)

    return output_image, status_md + warning, details_md, weather_md, conf_plot, sev_badge, audio_path, hindi_txt, report_file

def live_fn(webcam_frame):
    if webcam_frame is None:
        return None, "Waiting for camera ..."
    img_pil = Image.fromarray(webcam_frame) if isinstance(webcam_frame, np.ndarray) else webcam_frame
    img_array = preprocess_image(img_pil)
    predictions = model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(predictions))
    return webcam_frame, f"**{_class_name(idx)}** ({float(predictions[idx]):.1%})"

# ══════════════════════════════════════════════════════════════════
# 5. Launch UI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo = create_comprehensive_ui(predict_fn, live_fn)
    demo.launch()
