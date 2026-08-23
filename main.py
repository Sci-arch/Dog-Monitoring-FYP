import os
import io
import gc
import time
import glob
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from tensorflow.keras.models import load_model
from typing import Optional
import tensorflow as tf

# 1. 限制 TensorFlow 线程，防止卡死
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = FastAPI(title="FYP2 Multimodal Pet Emotion API")
latest_pet_status = {"diagnosed_emotion": "Waiting..."}

# 2. 全局加载模型
MODEL_PATH = "Cloud_Audio_CRNN_Advance.h5"

print("🚀 Loading model...")
if os.path.exists(MODEL_PATH):
    audio_model = load_model(MODEL_PATH)
    print("✅ Model loaded!")
else:
    audio_model = None
    print("❌ Model not found!")

def calculate_physio_risk(bpm: int, temp: float, action: int) -> float:
    risk = 0.0
    if bpm > 150: risk += 0.5
    elif bpm > 120: risk += 0.3
    if temp > 39.5: risk += 0.3
    if action in [1, 2]: risk += 0.2
    return min(risk, 1.0)

def cleanup_old_recordings(keep_last=10):
    files = sorted(glob.glob("debug_record_*.wav"), key=os.path.getmtime)
    if len(files) > keep_last:
        for old_file in files[:-keep_last]:
            try:
                os.remove(old_file)
            except:
                pass

@app.post("/analyze_emotion")
async def analyze_emotion(
    bpm: int = Form(...),
    temp: float = Form(...),
    action: int = Form(...),
    audio_file: Optional[UploadFile] = File(None)
):
    physio_risk = calculate_physio_risk(bpm, temp, action)
    audio_risk = 0.0
    saved_filename = None
    raw_ai_risk = None
    peak_volume = None
    rms_volume = None
    decision_source = "NO_AUDIO"
    fallback_used = False
    
    if audio_file is not None and audio_model is not None:
        try:
            audio_bytes = await audio_file.read()
            
            timestamp = int(time.time() * 1000)
            saved_filename = f"debug_record_{timestamp}.wav"
            with open(saved_filename, "wb") as f:
                f.write(audio_bytes)
            
            cleanup_old_recordings(keep_last=10)
            
            # 提取音频
            audio_io = io.BytesIO(audio_bytes)
            y, sr = librosa.load(audio_io, sr=16000)
            
            y_pre = np.append(y[0], y[1:] - 0.97 * y[:-1])
            S = librosa.feature.melspectrogram(y=y_pre, sr=sr, n_mels=128, fmax=8000, hop_length=160, win_length=400)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            if S_dB.shape[1] < 200:
                S_dB = np.pad(S_dB, pad_width=((0,0), (0, 200 - S_dB.shape[1])), mode='constant')
            else:
                S_dB = S_dB[:, :200]
            
            X_input = np.expand_dims(S_dB, axis=[0, -1])
            
            # ==========================================================
            # AUDIO AI + SIGNAL DIAGNOSTICS
            # ==========================================================
            audio_prediction = audio_model(X_input, training=False)
            raw_ai_risk = float(audio_prediction[0][0])

            peak_volume = float(np.max(np.abs(y)))
            rms_volume = float(np.sqrt(np.mean(np.square(y))))

            # ==========================================================
            # FINAL HYBRID AUDIO DECISION
            # CRNN is primary. Signal metrics are only conservative
            # fallbacks for silence or strong/uncertain input.
            # ==========================================================
            fallback_used = False

            if rms_volume < 0.02 and peak_volume < 0.05:
                audio_risk = 0.10
                decision_source = "SILENCE_FALLBACK"
                fallback_used = True

            elif (
                0.45 <= raw_ai_risk <= 0.60
                and peak_volume > 0.45
                and rms_volume > 0.08
            ):
                audio_risk = 0.85
                decision_source = "LOUD_SIGNAL_FALLBACK"
                fallback_used = True

            else:
                audio_risk = raw_ai_risk
                decision_source = "CRNN"

            print("========== AUDIO AI DEBUG ==========")
            print(f"Raw CRNN Risk   : {raw_ai_risk:.4f}")
            print(f"Peak Volume     : {peak_volume:.4f}")
            print(f"RMS Volume      : {rms_volume:.4f}")
            print(f"Final Audio Risk: {audio_risk:.4f}")
            print(f"Decision Source : {decision_source}")
            print(f"Fallback Used   : {fallback_used}")
            print("====================================")

            del audio_bytes, audio_io, y, y_pre, S, S_dB, X_input, audio_prediction
            gc.collect()
                
        except Exception as e:
            print(f"❌ Audio processing error: {e}")
    
    final_fusion_score = (0.6 * audio_risk) + (0.4 * physio_risk)
    
    if final_fusion_score > 0.8:
        emotion = "Class 1: Separation Anxiety (极度分离焦虑)"
    elif final_fusion_score > 0.6:
        emotion = "Class 2: Aggressive / Alert (警戒敌意)"
    elif final_fusion_score > 0.4:
        emotion = "Class 4: Physical Distress (身体异常/压力)"
    elif final_fusion_score > 0.2:
        emotion = "Class 0: Happy / Playful (兴奋玩耍)"
    else:
        emotion = "Class 3: Relaxed / Calm (静息放松)"
    
    global latest_pet_status
    latest_pet_status = {
        "status": "success",
        "diagnosed_emotion": emotion,
        "bpm": bpm,
        "temp": temp,
        "action": action,
        "raw_crnn_audio_risk": round(raw_ai_risk, 4) if raw_ai_risk is not None else None,
        "peak_volume": round(peak_volume, 4) if peak_volume is not None else None,
        "rms_volume": round(rms_volume, 4) if rms_volume is not None else None,
        "audio_risk": round(audio_risk, 4),
        "decision_source": decision_source,
        "fallback_used": fallback_used,
        "last_audio_file": saved_filename
    }

    return {
        "status": "success",
        "inputs_received": {"bpm": bpm, "temp": temp, "action": action},
        "audio_diagnostics": {
            "raw_crnn_risk": round(raw_ai_risk, 4) if raw_ai_risk is not None else None,
            "peak_volume": round(peak_volume, 4) if peak_volume is not None else None,
            "rms_volume": round(rms_volume, 4) if rms_volume is not None else None,
            "final_audio_risk": round(audio_risk, 4),
            "decision_source": decision_source,
            "fallback_used": fallback_used
        },
        "physio_risk_prob": round(physio_risk, 3),
        "final_fusion_score": round(final_fusion_score, 3),
        "diagnosed_emotion": emotion,
        "saved_audio": saved_filename
    }

@app.get("/latest_emotion")
async def get_latest_emotion():
    return latest_pet_status

@app.get("/list_audio")
async def list_audio():
    files = sorted(glob.glob("debug_record_*.wav"), key=os.path.getmtime, reverse=True)
    return {"total": len(files), "recordings": files}

@app.get("/download_latest")
async def download_latest():
    files = sorted(glob.glob("debug_record_*.wav"), key=os.path.getmtime, reverse=True)
    if files:
        return FileResponse(files[0], media_type="audio/wav")
    return JSONResponse({"error": "No recordings yet"}, status_code=404)

@app.get("/download_audio/{filename}")
async def download_specific(filename: str):
    if os.path.exists(filename) and filename.endswith(".wav"):
        return FileResponse(filename, media_type="audio/wav")
    return JSONResponse({"error": "File not found"}, status_code=404)

@app.get("/", response_class=HTMLResponse)
async def web_player():
    files = sorted(glob.glob("debug_record_*.wav"), key=os.path.getmtime, reverse=True)
    html_content = """
    <html>
    <head>
        <title>Roc's FYP Audio Player</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #1e1e1e; color: #fff; }
            .record-box { background: #2d2d2d; padding: 15px; margin-bottom: 15px; border-radius: 8px; }
            audio { width: 100%; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h2>🐕 FYP Multimodal Audio Logs (Latest 10)</h2>
    """
    if not files:
        html_content += "<p>No recordings found yet. Start the ESP32!</p>"
    else:
        for f in files:
            html_content += f"""
            <div class="record-box">
                <strong>{f}</strong>
                <audio controls src="/download_audio/{f}"></audio>
            </div>
            """
    html_content += "</body></html>"
    return html_content

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)