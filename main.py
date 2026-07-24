import os
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, Form
from tensorflow.keras.models import load_model
from typing import Optional

# 初始化 FastAPI 服务器 (必须在装饰器之前)
app = FastAPI(title="Roc's FYP2 Multimodal Pet Emotion API")
latest_pet_status = {"diagnosed_emotion": "Waiting..."}
# 1. 启动时加载你的左脑 (听觉专家 .h5 模型)
# MODEL_PATH = "Cloud_Audio_CRNN_Model.h5" 
MODEL_PATH = "Cloud_Audio_CRNN_Advance.h5" 

print("🚀 正在唤醒云端听觉专家...")
if os.path.exists(MODEL_PATH):
    audio_model = load_model(MODEL_PATH)
    print("✅ 左脑模型加载成功！")
else:
    audio_model = None  # 设置为 None，避免后面调用时报错
    print("❌ 找不到 .h5 模型，请确认它和 main.py 在同一个文件夹！")

# 2. 右脑：生理危险度映射公式
def calculate_physio_risk(bpm: int, temp: float, action: int) -> float:
    risk = 0.0
    # 心率映射 (假设正常心率是 80-120，超过 150 极度危险)
    if bpm > 150: 
        risk += 0.5
    elif bpm > 120: 
        risk += 0.3
    
    # 体温映射 (超过 39.5 度加危险分)
    if temp > 39.5: 
        risk += 0.3
    
    # 动作异常 (假设 1是狂奔, 2是抽搐)
    if action in [1, 2]: 
        risk += 0.2
    
    return min(risk, 1.0)  # 最高 100% (1.0)
# 3. 核心融合接口
@app.post("/analyze_emotion")
async def analyze_emotion(
    bpm: int = Form(...),
    temp: float = Form(...),
    action: int = Form(...),
    audio_file: Optional[UploadFile] = File(None)
):
    # --- A. 提取右脑生理危险度 ---
    physio_risk = calculate_physio_risk(bpm, temp, action)
    
    # --- B. 提取左脑声音危险度 ---
    audio_risk = 0.0 
    
    if audio_file is not None and audio_model is not None:
        try:
            audio_bytes = await audio_file.read()
            with open("temp.wav", "wb") as f:
                f.write(audio_bytes)
            
            y, sr = librosa.load("temp.wav", sr=16000)
            y_pre = np.append(y[0], y[1:] - 0.97 * y[:-1])
            S = librosa.feature.melspectrogram(y=y_pre, sr=sr, n_mels=128, fmax=8000, hop_length=160, win_length=400)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            if S_dB.shape[1] < 200:
                S_dB = np.pad(S_dB, pad_width=((0,0), (0, 200 - S_dB.shape[1])), mode='constant')
            else:
                S_dB = S_dB[:, :200]
            
            X_input = np.expand_dims(S_dB, axis=[0, -1])
            audio_prediction = audio_model.predict(X_input, verbose=0)
            audio_risk = float(audio_prediction[0][0])
            
            if os.path.exists("temp.wav"):
                os.remove("temp.wav")
                
        except Exception as e:
            print(f"❌ 音频处理出错: {e}")
    
    # --- C. Y-Path 终极 Late Fusion 大融合 ---
    final_fusion_score = (0.6 * audio_risk) + (0.4 * physio_risk)
    
    # --- D. 5 大情绪法官判决 ---
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
    
    # 🌟 修复点：在计算完 emotion 之后，把结果存进全局信箱！
    global latest_pet_status
    latest_pet_status = {
        "status": "success",
        "diagnosed_emotion": emotion
    }
    
    # 最后把详细结果返回给 ESP32
    return {
        "status": "success",
        "inputs_received": {"bpm": bpm, "temp": temp, "action": action},
        "audio_file_provided": audio_file is not None,
        "audio_risk_prob": round(audio_risk, 3),
        "physio_risk_prob": round(physio_risk, 3),
        "final_fusion_score": round(final_fusion_score, 3),
        "diagnosed_emotion": emotion
    }

# 专供手机读取的接口
@app.get("/latest_emotion")
async def get_latest_emotion():
    return latest_pet_status

# 🌟 必须加在最下面，让 Railway 自动分配端口
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)