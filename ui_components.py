"""
ui_components.py — Leaf Disease Detection Space
==================================================
Updated to support 9-output pipeline including:
 - Matplotlib Confidence Plots
 - Severity Badges
 - Audio generation (gTTS)
 - Hindi Translations
 - Downloadable Reports
 - Feedback System logging to CSV
"""

import gradio as gr
import csv
import datetime

# ── Inline CSS for nicer markdown tables ─────────────────────────────────────
_CUSTOM_CSS = """
#details-panel table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}
#details-panel th, #details-panel td {
    border: 1px solid var(--border-color-primary);
    padding: 6px 10px;
    text-align: left;
}
#details-panel th {
    background: var(--background-fill-secondary);
    font-weight: 600;
}
"""

_ABOUT_MD = """
## 🌿 Smart Leaf Disease Detection System

### How it works
1. **Upload** a clear photo of a plant leaf (or use your webcam).
2. **CLIP Gatekeeper** first checks that the image is actually a leaf.
3. **EfficientNetB3** classifies the leaf into one of 38 PlantVillage classes.
4. **Weather API** fetches real-time humidity & rain to adjust spray urgency.
5. **Pesticide table** lists recommended products, dosage, and timing.

### Crops covered
Apple · Blueberry · Cherry · Corn · Grape · Orange · Peach · 
Pepper · Potato · Raspberry · Soybean · Squash · Strawberry · Tomato

---
*Built by Git-Aditi03 | Purnea College of Engineering, Purnea*
"""

def log_feedback(feedback_text):
    """Saves user feedback directly to a local CSV file for review."""
    if not feedback_text.strip(): return "⚠️ Please enter feedback first."
    with open("feedback_log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), feedback_text])
    return "✅ Feedback logged to CSV! Thank you."


def create_comprehensive_ui(predict_fn, live_fn):
    with gr.Blocks(
        title="🌿 Advanced Leaf Disease Detector",
        theme=gr.themes.Soft(primary_hue="emerald"),
        css=_CUSTOM_CSS,
    ) as demo:

        gr.Markdown(
            "<h1 style='text-align:center;color:#047857;'>"
            "🌿 Smart Leaf Disease Detection System</h1>"
            "<h3 style='text-align:center;'>Developed by Git-Aditi03 | Purnea College of Engineering, Purnea</h3>"
        )
        
        with gr.Tabs():

            # ══════════════════════════════════════════════════════
            # TAB 1 — Full image analysis pipeline
            # ══════════════════════════════════════════════════════
            with gr.Tab("📸 Image Analysis & Diagnostics"):
                with gr.Row():
                    
                    # ── Left column: Inputs ───────────────────────
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="pil", 
                            label="Upload Leaf Image", 
                            sources=["upload"], 
                            height=350
                        )
                        location_input = gr.Textbox(
                            value="Purnea, Bihar", 
                            label="📍 Your City (for Weather & Risk Analysis)"
                        )
                        gradcam_check = gr.Checkbox(
                            value=True, 
                            label="Apply Disease Heatmap (Grad-CAM)"
                        )
                        analyze_btn = gr.Button("🔍 Analyze Leaf & Generate Report", variant="primary", size="lg")
                        
                    # ── Right column: Primary Outputs ──────────────
                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Processed Image Output")
                        severity_output = gr.Markdown(label="🚨 Severity Score")
                        status_output = gr.Markdown(label="📊 Pipeline Status & Diagnosis")
                        weather_output = gr.Markdown(label="🌤️ Real-Time Weather & Spread Risk")
                        
                with gr.Row():
                    with gr.Column(scale=1):
                        plot_output = gr.Plot(label="Confidence Levels Breakdown")
                        
                    with gr.Column(scale=1):
                        audio_output = gr.Audio(label="🔊 Voice Output (English)")
                        report_output = gr.File(label="📄 Download Diagnostic Report (.txt)")
                        
                with gr.Row():
                    with gr.Column(scale=1):
                        details_output = gr.Markdown(
                            label="🧪 Detailed Treatment & Pesticide Info",
                            elem_id="details-panel"
                        )
                    with gr.Column(scale=1):
                        hindi_output = gr.Markdown(label="🇮🇳 Hindi Translation Summary")
                        
                        gr.Markdown("### 📝 Submit Feedback")
                        feedback_in = gr.Textbox(label="Did this help? Any issues?", lines=2)
                        feedback_btn = gr.Button("Submit Feedback")
                        feedback_status = gr.Markdown()
                        feedback_btn.click(fn=log_feedback, inputs=feedback_in, outputs=feedback_status)

                # Bind all 9 outputs sequentially to the updated predict_fn
                analyze_btn.click(
                    fn=predict_fn,
                    inputs=[image_input, location_input, gradcam_check],
                    outputs=[
                        output_image, 
                        status_output, 
                        details_output, 
                        weather_output, 
                        plot_output, 
                        severity_output, 
                        audio_output, 
                        hindi_output, 
                        report_output
                    ]
                )

            # ══════════════════════════════════════════════════════
            # TAB 2 — Real-time webcam
            # ══════════════════════════════════════════════════════
            with gr.Tab("📹 Live Scanner"):
                gr.Markdown(
                    "### Rapid Status Check\n"
                    "> ⚡ **Advanced analytics (weather, audio, plots) are disabled** "
                    "here for maximum real-time speed. Use the Image Analysis tab for full results."
                )
                with gr.Row():
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="Live Webcam Feed",
                        height=400,
                        type="numpy",
                    )
                    with gr.Column():
                        live_image = gr.Image(label="Live Output Frame")
                        live_status = gr.Markdown(label="⚡ Live Model Status")

                webcam_input.stream(
                    fn=live_fn,
                    inputs=webcam_input,
                    outputs=[live_image, live_status],
                    stream_every=1.0,
                )

            # ══════════════════════════════════════════════════════
            # TAB 3 — About / Model info
            # ══════════════════════════════════════════════════════
            with gr.Tab("ℹ️ About"):
                gr.Markdown(_ABOUT_MD)

    return demo
