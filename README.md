# 🐾 Wearable Pet Monitoring System with Edge-Cloud AI 

![Project Status](https://img.shields.io/badge/Status-Active_Development-brightgreen)
![Hardware](https://img.shields.io/badge/Hardware-ESP32%20%7C%20MPU6500%20%7C%20MAX30102-blue)
![AI](https://img.shields.io/badge/AI-TinyML%20%7C%201D--CNN%20%7C%20Multimodal-orange)

## 📌 Project Overview
This project proposes a dual-path Artificial Intelligence of Things (AIoT) wearable system designed to monitor canine health and emotional well-being in real time. 

By integrating **TinyML** for low-latency physical anomaly detection on the edge (ESP32) and a **Cloud-based Multimodal CRNN** for complex acoustic emotion classification, the system solves the Context Alignment Problem without relying on privacy-invasive cameras.

## ⚙️ Architecture Highlights
1. **Edge AI Path (Local Actuation):** - Sensors: MPU6500 (IMU) + MAX30102 (PPG).
   - Logic: 1D-CNN (INT8 Quantized) running locally on the ESP32 via a 100-step sliding window buffer.
   - Output: Real-time physical stress/fall detection (< 100ms latency).
2. **Cloud AI Path (Multimodal Fusion):** - Sensor: INMP441 (I2S Microphone).
   - Logic: Trigger-based audio recording sent via MQTT JSON payloads to AWS. Processed using Log-Mel Spectrograms and Late Fusion layers.
3. **Fail-Safe Path:** Local DS18B20 temperature thresholding for immediate fever alerts.

## 📂 Repository Structure
- `/Arduino_Edge` - Firmware for the ESP32 (Sensor routing, TinyML invocation, MQTT).
- `/Colab_Models` - Jupyter notebooks for training the 1D-CNN and cloud models.
- `/Mobile_App` - (Upcoming) Android dashboard source code.
- `model.h` - The quantized C++ header file for edge deployment.

## 📊 Datasets Used
*Note: Due to copyright and size limitations, original datasets are not hosted in this repository.*
1. **Canine Kinetic Data:** Trained using the open-source Dog Behavior Analysis Dataset. [Link to Kaggle/Source]
2. **Physiological Data:** PPG signals normalized from the PPG-DaLiA framework.
3. **Acoustic Benchmark:** Canine vocalizations utilizing the Mescalina database. Please request access from the original authors.

## 🛠️ How to Use
1. Clone this repository.
2. Ensure you have the `TensorFlowLite_ESP32` and `MPU6500_WE` libraries installed in your Arduino IDE.
3. Upload the `.ino` sketch to your ESP32 board. (Note: Avoid using GPIO 2 and 5 during the upload sequence).
