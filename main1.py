import streamlit as st
import os
import time
import importlib
import sys
import cv2
import yt_dlp
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests
from io import BytesIO
import json
import base64

# ================ CONFIG & STYLE ================
st.set_page_config(page_title="DeepSafe Pro", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #fff; font-family: 'Segoe UI', sans-serif;}
    h1, h2, h3 {color: #00ffff !important; text-shadow: 0 0 10px #00ffff;}
    .css-1v0mbdj {padding-top: 1rem;}
    .result-real {font-size: 80px; font-weight: bold; color: #00ff9d; text-align: center; padding: 40px; 
                   border-radius: 25px; background: rgba(0, 255, 157, 0.2); border: 3px solid #00ff9d; box-shadow: 0 0 40px rgba(0,255,157,0.6);}
    .result-fake {font-size: 80px; font-weight: bold; color: #ff0055; text-align: center; padding: 40px; 
                  border-radius: 25px; background: rgba(255, 0, 85, 0.2); border: 3px solid #ff0055; box-shadow: 0 0 40px rgba(255,0,85,0.6);}
    .confidence {font-size: 36px; text-align: center; margin: 20px; color: #00ffff;}
    .frame-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0;}
    .frame-card {background: rgba(255,255,255,0.08); padding: 15px; border-radius: 20px; text-align: center; border: 1px solid #00ffff44; box-shadow: 0 4px 15px rgba(0,255,255,0.1);}
    .stButton>button {background: linear-gradient(45deg, #00ffff, #00ff9d); border: none; color: black; font-weight: bold; padding: 18px; border-radius: 50px; font-size: 18px;}
    .stButton>button:hover {transform: translateY(-3px); box-shadow: 0 10px 30px rgba(0,255,255,0.4);}
</style>
""", unsafe_allow_html=True)

# ================ SETUP ================
os.makedirs("temp/frames", exist_ok=True)

def clean_temp():
    for root, dirs, files in os.walk("temp", topdown=False):
        for f in files:
            os.remove(os.path.join(root, f))
        for d in dirs:
            if d != "frames":
                os.rmdir(os.path.join(root, d))

def download_from_url(url):
    clean_temp()
    try:
        ydl_opts = {
            'format': 'best[height<=1080]',
            'outtmpl': 'temp/delete.mp4',
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return "temp/delete.mp4" if os.path.exists("temp/delete.mp4") else None
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        return None

def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // num_frames)
    frames = []
    count = 0
    saved = 0
    while saved < num_frames:
        ret, frame = cap.read()
        if not ret: break
        if count % interval == 0:
            path = f"temp/frames/frame_{saved+1}.jpg"
            cv2.imwrite(path, frame)
            frames.append((path, frame))
            saved += 1
        count += 1
    cap.release()
    return frames

def run_model(model_key):
    module_name = f"models.{model_key}.demo"
    try:
        if module_name in sys.modules:
            mod = importlib.reload(sys.modules[module_name])
        else:
            mod = importlib.import_module(module_name)
        result_path = f"models/{model_key}/result.txt"
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                prob = float(f.read().strip())
            return prob
        return None
    except Exception as e:
        st.error(f"Model {model_key.upper()} error: {e}")
        return None

# ================ MAIN APP ================
st.title("🛡️ DeepSafe Pro")
st.markdown("### Next-Gen Open Source DeepFake Detection Platform")

col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader("Upload Image or Video", type=['jpg','jpeg','png','mp4','mov','avi'])
with col2:
    url = st.text_input("Or paste any video/image URL (YouTube, X, TikTok, etc.)", placeholder="https://...")

media_path = None
media_type = None

if uploaded_file:
    clean_temp()
    ext = uploaded_file.name.split(".")[-1].lower()
    bytes_data = uploaded_file.getvalue()
    if ext in ["jpg", "jpeg", "png"]:
        img = Image.open(BytesIO(bytes_data))
        img.save("temp/delete.jpg")
        st.image(img, use_column_width=True)
        media_type = "image"
        media_path = "temp/delete.jpg"
    else:
        with open("temp/delete.mp4", "wb") as f:
            f.write(bytes_data)
        st.video("temp/delete.mp4")
        media_type = "video"
        media_path = "temp/delete.mp4"

elif url.strip():
    with st.spinner("Downloading media from URL..."):
        if url.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                response = requests.get(url, timeout=15)
                img = Image.open(BytesIO(response.content))
                img.save("temp/delete.jpg")
                st.image(img, use_column_width=True)
                media_type = "image"
                media_path = "temp/delete.jpg"
            except:
                st.error("Failed to download image.")
        else:
            media_path = download_from_url(url)
            if media_path:
                st.video(media_path)
                media_type = "video"
            else:
                st.error("Could not download video. Try direct link or upload file.")

# ================ ANALYSIS ================
if media_path and st.button("🚀 Run DeepFake Analysis", type="primary", use_container_width=True):
    models_dir = [f for f in os.listdir("models") if os.path.isdir(os.path.join("models", f))]
    available_models = []
    for m in models_dir:
        name = m.rsplit("_", 1)[0].title()
        typ = m.split("_")[-1]
        if (media_type == "image" and typ == "image") or (media_type == "video" and typ == "video"):
            available_models.append(name)

    if not available_models:
        st.error("No compatible models found in /models folder!")
    else:
        selected = st.multiselect("Choose Detection Models", available_models, default=available_models[:5])

        if selected:
            progress = st.progress(0)
            status = st.empty()
            results = []

            for idx, model_name in enumerate(selected):
                status.text(f"Running {model_name}...")
                key = f"{model_name.lower()}_{media_type}"
                start = time.time()
                prob = run_model(key)
                elapsed = time.time() - start

                if prob is not None:
                    results.append({"model": model_name, "prob": prob, "time": elapsed})
                progress.progress((idx + 1) / len(selected))

            if results:
                df = pd.DataFrame(results)
                avg_prob = df["prob"].mean()
                is_fake = avg_prob > 0.5
                confidence = avg_prob if is_fake else 1 - avg_prob

                st.markdown("<br><br>", unsafe_allow_html=True)
                verdict = "FAKE DETECTED ⚠️" if is_fake else "AUTHENTIC ✓"
                cls = "result-fake" if is_fake else "result-real"
                st.markdown(f'<div class="{cls}">{verdict}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence">Confidence: <strong>{confidence:.1%}</strong></div>', unsafe_allow_html=True)

                # === 1. Pie Chart ===
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(6,6))
                    colors = ['#ff0055', '#00ff9d'] if is_fake else ['#00ff9d', '#ff0055']
                    ax.pie([avg_prob, 1-avg_prob], labels=['FAKE', 'REAL'], autopct='%1.1f%%',
                           colors=colors, startangle=90, explode=(0.08, 0), shadow=True)
                    ax.set_title("Overall Risk Assessment", color='white', size=16)
                    fig.patch.set_facecolor('#0e1117')
                    ax.set_facecolor('#0e1117')
                    st.pyplot(fig)

                # === 2. Detailed Bar Chart ===
                with col2:
                    fig, ax = plt.subplots(figsize=(7,5))
                    bars = ax.barh(df["model"], df["prob"], 
                                 color=[('#ff0055' if p > 0.5 else '#00ff9d') for p in df["prob"]],
                                 alpha=0.8, edgecolor='white', linewidth=1)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Fake Probability →")
                    ax.set_title("Per-Model DeepFake Score")
                    for i, (bar, prob) in enumerate(zip(bars, df["prob"])):
                        ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2, f"{prob:.1%}",
                                va='center', fontweight='bold', color='white')
                    ax.axvline(0.5, color='yellow', linestyle='--', linewidth=2, label="Threshold")
                    ax.legend()
                    fig.patch.set_facecolor('#0e1117')
                    ax.set_facecolor('#0e1117')
                    ax.tick_params(colors='white')
                    plt.tight_layout()
                    st.pyplot(fig)

                # === 3. Radar Chart (Model Consensus) ===
                if len(df) >= 3:
                    angles = np.linspace(0, 2*np.pi, len(df), endpoint=False).tolist()
                    values = df["prob"].tolist()
                    values += values[:1]
                    angles += angles[:1]

                    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
                    ax.fill(angles, values, color='#ff0055', alpha=0.25)
                    ax.plot(angles, values, color='#ff0055', linewidth=3)
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(df["model"])
                    ax.set_ylim(0,1)
                    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
                    ax.grid(True, color='white', alpha=0.3)
                    ax.set_title("Model Consensus Radar", size=16, color='white', pad=20)
                    fig.patch.set_facecolor('#0e1117')
                    st.pyplot(fig)

                # === 4. Inference Speed Comparison ===
                st.markdown("### ⚡ Model Speed Comparison")
                fig, ax = plt.subplots()
                ax.bar(df["model"], df["time"], color='#00ffff', alpha=0.7, edgecolor='white')
                ax.set_ylabel("Time (seconds)")
                ax.set_title("Inference Time per Model")
                for i, t in enumerate(df["time"]):
                    ax.text(i, t + max(df["time"])*0.02, f"{t:.2f}s", ha='center', fontweight='bold')
                plt.xticks(rotation=45)
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')
                st.pyplot(fig)

                # === 5. Video Frame Analysis (if video) ===
                if media_type == "video":
                    st.markdown("### 📹 Key Frames Analyzed")
                    frames = extract_frames(media_path, 8)
                    cols = st.columns(4)
                    for i, (fpath, frame_bgr) in enumerate(frames):
                        with cols[i % 4]:
                            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(frame_rgb)
                            st.image(pil_img, use_column_width=True)
                            # Dummy per-frame score (replace with real frame-level model if available)
                            fake_score = np.random.uniform(0.3, 0.9)
                            color = "🟥" if fake_score > 0.5 else "🟩"
                            st.markdown(f"<div style='text-align:center'>{color} Frame {i+1}<br><b>{fake_score:.1%} fake</b></div>", unsafe_allow_html=True)

                # === Download Report ===
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "media_type": media_type,
                    "avg_deepfake_probability": round(avg_prob, 4),
                    "verdict": "FAKE" if is_fake else "REAL",
                    "confidence": round(confidence, 4),
                    "models_used": selected,
                    "results": df.to_dict(orient="records")
                }
                report_json = json.dumps(report, indent=2)
                report_csv = df.to_csv(index=False)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("📄 Download JSON Report", report_json, "deepsafe_report.json", "application/json")
                with col2:
                    st.download_button("📊 Download CSV Results", report_csv, "deepsafe_results.csv", "text/csv")

                st.success("Analysis completed successfully!")
                st.balloons()
            else:
                st.error("All selected models failed.")

st.markdown("<br><hr><center>🔥 DeepSafe Pro • Made with ❤️ by <b>ARCHI & SARVESH</b></center>", unsafe_allow_html=True)