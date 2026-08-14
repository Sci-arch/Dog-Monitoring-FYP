import os
import io
import gc
import time
import glob
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from tensorflow.keras.models import load_model
from typing import Optional
import tensorflow as tf

# 1. Limit TensorFlow threads
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = FastAPI(title="FYP2 Multimodal Pet Emotion API")
latest_pet_status = {"diagnosed_emotion": "Waiting..."}

# 2. Load model once
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
    """Delete old recordings, keep only latest N files"""
    files = sorted(glob.glob("debug_record_*.wav"), key=os.path.getmtime)
    if len(files) > keep_last:
        for old_file in files[:-keep_last]:
            try:
                os.remove(old_file)
                print(f"🗑️ Deleted old: {old_file}")
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
    
    if audio_file is not None and audio_model is not None:
        try:
            audio_bytes = await audio_file.read()
            
            # Save with unique timestamp
            timestamp = int(time.time() * 1000)
            saved_filename = f"debug_record_{timestamp}.wav"
            with open(saved_filename, "wb") as f:
                f.write(audio_bytes)
            print(f"💾 Audio saved: {saved_filename}")
            
            # Auto-cleanup (keep last 10)
            cleanup_old_recordings(keep_last=10)
            
            # Process audio
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
            
            audio_prediction = audio_model(X_input, training=False)
            audio_risk = float(audio_prediction[0][0])
            
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
        "last_audio_file": saved_filename
    }
    
    return {
        "status": "success",
        "inputs_received": {"bpm": bpm, "temp": temp, "action": action},
        "audio_risk_prob": round(audio_risk, 3),
        "physio_risk_prob": round(physio_risk, 3),
        "final_fusion_score": round(final_fusion_score, 3),
        "diagnosed_emotion": emotion,
        "saved_audio": saved_filename
    }

@app.get("/latest_emotion")
async def get_latest_emotion():
    return latest_pet_status

# 📋 List all recordings
@app.get("/list_audio")
async def list_audio():
    files = sorted(glob.glob("debug_record_*.wav"), key=os.path.getmtime, reverse=True)
    return {
        "total": len(files),
        "recordings": files
    }

# ⬇️ Download latest recording (quick access)
@app.get("/download_latest")
async def download_latest():
    files = sorted(glob.glob("debug_record_*.wav"), key=os.path.getmtime, reverse=True)
    if files:
        return FileResponse(files[0], media_type="audio/wav")
    return JSONResponse({"error": "No recordings yet"}, status_code=404)

# ⬇️ Download specific recording by filename
@app.get("/download_audio/{filename}")
async def download_specific(filename: str):
    if os.path.exists(filename) and filename.endswith(".wav"):
        return FileResponse(filename, media_type="audio/wav")
    return JSONResponse({"error": "File not found"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)