# 🐾 Wearable Pet Monitoring System with Edge-Cloud AI

![Project Status](https://img.shields.io/badge/Status-Active_Development-brightgreen)
![Hardware](https://img.shields.io/badge/Hardware-ESP32%20%7C%20MPU6500%20%7C%20MAX30102-blue)
![AI](https://img.shields.io/badge/AI-TinyML%20%7C%201D--CNN%20%7C%20Multimodal-orange)

## 📌 Project Overview
This project proposes a dual-path Artificial Intelligence of Things (AIoT) wearable system designed to monitor canine health and emotional well-being in real time.

By integrating **TinyML** for low-latency physical anomaly detection on the edge (ESP32) and a **Cloud-based Multimodal CRNN** for complex acoustic emotion classification, the system solves the Context Alignment Problem without relying on privacy-invasive cameras.

---

## 📂 Repository Navigation (Click to Jump)

| Module | Description | Directory |
| :--- | :--- | :--- |
| **⚙️ Edge Firmware** | ESP32 C++ code for sensor polling (I2C, I2S), TinyML inference, and MQTT. | [`/ESP32_Code`](./ESP32_Code) |
| **📱 Mobile App** | Android Kotlin application for UI visualization (Live graphs, Alert History). | [`/Android_App`](./Android_App) |
| **📊 Datasets Info** | Links to Kaggle and INAOE acoustic datasets used for training. | [`/Datasets`](./Datasets) |
| **☁️ Cloud Backend** | FastAPI server files hosted on Render. (Root directory). | `main.py` / `requirements.txt` |

---

## ⚙️ Architecture Highlights
1. **Edge AI Path (Local Actuation):**
   - Sensors: MPU6500 (IMU) + MAX30102 (PPG).
   - Logic: 1D-CNN (INT8 Quantized) running locally on the ESP32.
   - Output: Real-time physical stress/fall detection (< 100ms latency).
2. **Cloud AI Path (Multimodal Fusion):**
   - Sensor: INMP441 (I2S Microphone).
   - Logic: Log-Mel Spectrograms processed by a CRNN with Late Fusion.
3. **Fail-Safe Path:** DS18B20 temperature thresholding for immediate fever alerts.

---

## 🛠️ How to Deploy
1. **Cloud (Render):** Connect this repository to Render. It will automatically detect `main.py` and `requirements.txt` in the root folder.
2. **Edge (ESP32):** Open the `.ino` file from the `/ESP32_Code` folder in Arduino IDE. Ensure `TensorFlowLite_ESP32` and `MPU6500_WE` libraries are installed.
3. **Mobile:** Open the project in `/Android_App` using Android Studio.

---

## 📊 Datasets Used
*Note: Due to size limitations, raw datasets are stored on Google Drive.*
- **Motion:** Dog Behavior Analysis Dataset (Kaggle).
- **Physiological:** PPG-DaLiA Dataset.
- **Acoustic:** Mescalina 2017 Database (Acquired via author agreement).

---

## 👨‍💻 Author
**Tang Jia Jian** - Final Year Project, UTAR (Kampar), 2026.
