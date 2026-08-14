import os
import io
import gc
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse  # 🌟 新增：用来下载文件
from tensorflow.keras.models import load_model
from typing import Optional
import tensorflow as tf

# 1. 限制 TensorFlow 线程
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = FastAPI(title="FYP2 Multimodal Pet Emotion API")
latest_pet_status = {"diagnosed_emotion": "Waiting..."}

# 2. 全局只加载一次模型
MODEL_PATH = "Cloud_Audio_CRNN_Advance.h5" 

print("🚀 正在唤醒云端听觉专家...")
if os.path.exists(MODEL_PATH):
    audio_model = load_model(MODEL_PATH)
    print("✅ 左脑模型加载成功！")
else:
    audio_model = None
    print("❌ 找不到 .h5 模型！")

def calculate_physio_risk(bpm: int, temp: float, action: int) -> float:
    risk = 0.0
    if bpm > 150: risk += 0.5
    elif bpm > 120: risk += 0.3
    if temp > 39.5: risk += 0.3
    if action in [1, 2]: risk += 0.2
    return min(risk, 1.0)

@app.post("/analyze_emotion")
async def analyze_emotion(
    bpm: int = Form(...),
    temp: float = Form(...),
    action: int = Form(...),
    audio_file: Optional[UploadFile] = File(None)
):
    physio_risk = calculate_physio_risk(bpm, temp, action)
    audio_risk = 0.0 
    
    # ✅ 修复了缩进！
    if audio_file is not None and audio_model is not None:
        try:
            audio_bytes = await audio_file.read()
            
            # 👇 保存在 Render 云端本地
            with open("debug_record.wav", "wb") as f:
                f.write(audio_bytes)
            print("💾 音频已保存为 debug_record.wav")
            
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
            print(f"❌ 音频处理出错: {e}")
    
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
        "temp": temp
    }
    
    return {
        "status": "success",
        "inputs_received": {"bpm": bpm, "temp": temp, "action": action},
        "audio_risk_prob": round(audio_risk, 3),
        "physio_risk_prob": round(physio_risk, 3),
        "final_fusion_score": round(final_fusion_score, 3),
        "diagnosed_emotion": emotion
    }

@app.get("/latest_emotion")
async def get_latest_emotion():
    return latest_pet_status

# 🌟 超级黑科技接口：用来下载最新的录音！
@app.get("/download_audio")
async def download_audio():
    if os.path.exists("debug_record.wav"):
        return FileResponse("debug_record.wav", media_type="audio/wav")
    return {"error": "No audio recorded yet!"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)