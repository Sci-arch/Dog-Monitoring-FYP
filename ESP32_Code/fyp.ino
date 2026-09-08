/*
 * Wearable Pet Monitoring System - LOCAL EDGE API TEST
 * Features: MPU6500, MAX30102, DS18B20, INMP441, OLED, Buzzer, TinyML
 * Network: WiFi + HTTP POST (Direct to Render Cloud API)
 * 
 * ⚠️ CRITICAL: Server endpoint must include /analyze_emotion suffix
 * 
 * Architecture Note (Pending Snapshot Mechanism):
 * Core 1 continuously samples sensors (IMU, Temp) every 20ms.
 * During the ~7s network POST wait time on Core 0, Core 1 continues building new TinyML windows.
 * The "Pending Snapshot" design ensures the ESP32 overwrites the payload buffer with the 
 * absolute latest data matrix rather than queueing old frames, preventing memory fragmentation.
 */

#include <WiFi.h>
#include <HTTPClient.h> 
#include <WiFiClientSecure.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <MAX30105.h>
#include "heartRate.h"         
#include <ArduinoJson.h>       
#include <driver/i2s.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <MPU6500_WE.h> 
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h" 
#include "tensorflow/lite/micro/micro_error_reporter.h" 
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h" 

// Logging Modes:
// 1 = Normal
// 2 = Serial Plotter
// 3 = Debug
#define LOG_MODE 2

// BPM detailed debug:
// 0 = clean BPM output
// 1 = detailed beat detection
#define DEBUG_BPM 1

// ============================================================
//  NETWORK CONFIGURATION
// ============================================================
const char* ssid = "GalaxyAa";        
const char* password = "cybercrime"; 
const char* serverName = "https://dog-api-lttt.onrender.com/analyze_emotion";

// ============================================================
//  HARDWARE PIN DEFINITIONS
// ============================================================
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

MAX30105 particleSensor;
MPU6500_WE imu = MPU6500_WE(0x68);

#define I2S_PORT I2S_NUM_0
// Optimized I2S pin configuration for stability
#define I2S_SCK  18
#define I2S_WS   19
#define I2S_SD   23

#define ONE_WIRE_BUS 15
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

#define LED_PIN     2       
#define BUZZER_PIN  5       

// ============================================================
//  BPM PROCESSING - IMPROVED ALGORITHM
// ============================================================
const byte RATE_SIZE = 4;
byte rates[RATE_SIZE] = {0};
byte rateSpot = 0;
long lastBeat = 0;
float beatsPerMinute = 0.0;
int beatAvg = 0;

// Finger detection threshold - based on your actual IR readings
// Your tests showed: No finger ~700-900, Finger ~70,000 → 215,000
const long BPM_IR_THRESHOLD = 10000;

// If no heartbeat detected for this long, BPM is considered stale
const unsigned long BPM_TIMEOUT_MS = 5000;

// ============================================================
//  SIMPLE REAL-TIME BPM DETECTOR - ADD THESE GLOBALS
// ============================================================
float irBaseline = 0.0;
float irFiltered = 0.0;
float irPrevious = 0.0;

unsigned long lastPulseTime = 0;
unsigned long lastValidPulseTime = 0;

float realtimeBPM = 0.0;

const float BASELINE_ALPHA = 0.01;
const float SIGNAL_ALPHA = 0.25;

// Adaptive peak threshold
const float MIN_PULSE_AMPLITUDE = 300.0;

// Physiological limits
const unsigned long MIN_BEAT_INTERVAL = 300;   // 200 BPM
const unsigned long MAX_BEAT_INTERVAL = 2000;  // 30 BPM

bool pulseWasRising = false;

unsigned long beatDetectedCount = 0;

// ============================================================
//  TINYML CONFIGURATION
// ============================================================
constexpr int kTensorArenaSize = 60 * 1024; 
uint8_t tensor_arena[kTensorArenaSize];
const tflite::Model* tflite_model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input_tensor;
TfLiteTensor* output_tensor;
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

// ============================================================
//  SENSOR BUFFERS
// ============================================================
#define TIME_STEPS 100
#define FEATURES 3  
float sensor_buffer[TIME_STEPS][FEATURES];
int buffer_index = 0; 
int current_action_id = 0; 
float tempC = 0.0;

// IIR Filter Coefficients
float b0 = 0.067455, b1 = 0.13491, b2 = 0.067455;
float a1 = -1.14298, a2 = 0.41280;
float x_hist[3][2] = {{0,0}, {0,0}, {0,0}}; 
float y_hist[3][2] = {{0,0}, {0,0}, {0,0}};

unsigned long lastTempReadTime = 0;
unsigned long lastLoopTime = 0;

// ============================================================
//  DUAL-CORE SYNCHRONIZATION (Core 0 = network, Core 1 = sensors)
// ============================================================
struct SensorSnapshot {
  uint32_t sequence;      
  unsigned long timestamp; 
  int bpm;
  float temp;
  int action;
};

// Function prototypes declaration
bool sendDataToCloud(SensorSnapshot snap);
void parseServerResponse(String responseStr, uint32_t sequence, int32_t micMax, bool micSignal, int httpStatus, unsigned long latency, size_t payloadSize);
bool evaluateAlerts();
void actuateHardware(bool criticalAlert, unsigned long currentMillis);
void updateDisplay(bool criticalAlert);
void networkTaskFunc(void *pvParameters);
void updateBPM(long irValue);

SensorSnapshot pendingSnapshot;
volatile bool pendingValid = false;
uint32_t snapshotSequence = 0;
portMUX_TYPE snapshotMux = portMUX_INITIALIZER_UNLOCKED;

volatile bool networkBusy = false;

enum NetworkState {
  NET_IDLE,
  NET_RECORDING,
  NET_SENDING,
  NET_WAITING,
  NET_SUCCESS,
  NET_ERROR
};
volatile NetworkState networkState = NET_IDLE;
TaskHandle_t networkTaskHandle = NULL;
SemaphoreHandle_t serialMux;

unsigned long lastSensorPrintTime = 0;
const unsigned long SENSOR_PRINT_INTERVAL_MS = 1000;

// OLED separation
unsigned long lastDisplayUpdate = 0;
const unsigned long DISPLAY_INTERVAL_MS = 500;

// ============================================================
//  MICROPHONE DIAGNOSTICS
// ============================================================
volatile bool micRecording  = false;
volatile bool micHasSignal  = false;
volatile int32_t micMaxAbs  = 0;
volatile int32_t micAvgAbs  = 0;
const int32_t MIC_SIGNAL_THRESHOLD = 300;

// ============================================================
//  LOGGING HELPERS (Mutex-guarded to prevent interleave)
// ============================================================
void logSensor(uint32_t seq, long ir, int bpm, float temp, int action, float accX, float accY, float accZ) {
  if (xSemaphoreTake(serialMux, pdMS_TO_TICKS(50)) != pdTRUE) return;
  
  #if LOG_MODE == 1
    Serial.printf("[SENSOR] #%u | BPM:%d | Temp:%.2f | Action:%d\n", seq, bpm, temp, action);
  #elif LOG_MODE == 2
    // Compatible with Serial Monitor and Serial Plotter
    Serial.printf("IR:%ld, BPM:%d, Temp:%.2f, Action:%d, AccX:%.2f, AccY:%.2f, AccZ:%.2f\n", ir, bpm, temp, action, accX, accY, accZ);
  #elif LOG_MODE == 3
    Serial.printf("[SENSOR] #%u | BPM:%d | Temp:%.2f | Action:%d | AccX:%.2f AccY:%.2f AccZ:%.2f\n", seq, bpm, temp, action, accX, accY, accZ);
  #endif
  
  xSemaphoreGive(serialMux);
}

void logNet(uint32_t seq, const char* status) {
  if (xSemaphoreTake(serialMux, pdMS_TO_TICKS(50)) == pdTRUE) {
    Serial.printf("[NET] #%u | %s\n", seq, status);
    xSemaphoreGive(serialMux);
  }
}

// ============================================================
//  FILTER & WAV GENERATOR
// ============================================================
float apply_filter(int axis, float raw_val) {
  float filtered = b0 * raw_val + b1 * x_hist[axis][0] + b2 * x_hist[axis][1] 
                   - a1 * y_hist[axis][0] - a2 * y_hist[axis][1];
  x_hist[axis][1] = x_hist[axis][0]; 
  x_hist[axis][0] = raw_val;
  y_hist[axis][1] = y_hist[axis][0]; 
  y_hist[axis][0] = filtered;
  return filtered;
}

void createWavHeader(byte* header, int waveDataSize) {
  header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
  unsigned int fileSize = waveDataSize + 36;
  header[4] = (byte)(fileSize & 0xFF); 
  header[5] = (byte)((fileSize >> 8) & 0xFF); 
  header[6] = (byte)((fileSize >> 16) & 0xFF); 
  header[7] = (byte)((fileSize >> 24) & 0xFF);
  header[8] = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';
  header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
  header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0;
  header[20] = 1; header[21] = 0;
  header[22] = 1; header[23] = 0;
  header[24] = 0x80; header[25] = 0x3E; header[26] = 0x00; header[27] = 0x00; 
  header[28] = 0x00; header[29] = 0x7D; header[30] = 0x00; header[31] = 0x00; 
  header[32] = 2; header[33] = 0;
  header[34] = 16; header[35] = 0; 
  header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
  header[40] = (byte)(waveDataSize & 0xFF); 
  header[41] = (byte)((waveDataSize >> 8) & 0xFF); 
  header[42] = (byte)((waveDataSize >> 16) & 0xFF); 
  header[43] = (byte)((waveDataSize >> 24) & 0xFF);
}

// ============================================================
//  WIFI SETUP (Hang prevention implemented)
// ============================================================
void setup_wifi() {
  delay(10);
  Serial.println();
  
  // Core Fix: Clear previous WiFi configurations and set to STA mode to prevent hangs
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true); 
  delay(1000); 

  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  
  int retries = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    retries++;
    if (retries > 30) { 
      Serial.println("\n[ERROR] WiFi module unresponsiveness detected. Initiating system restart...");
      delay(1000);
      ESP.restart(); 
    }
  }
  Serial.println("\nWiFi connected! IP: ");
  Serial.println(WiFi.localIP());
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  serialMux = xSemaphoreCreateMutex();
  setup_wifi(); 

  Wire.begin(21, 22); 
  Wire.setClock(400000); // Faster I2C for MAX30102
  
  pinMode(LED_PIN, OUTPUT); 
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW); 
  digitalWrite(BUZZER_PIN, HIGH); 
  
  // ===== I2S Microphone Setup =====
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false
  };
  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK, 
    .ws_io_num = I2S_WS, 
    .data_out_num = I2S_PIN_NO_CHANGE, 
    .data_in_num = I2S_SD
  };
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);

  // ===== TinyML Setup =====
  tflite_model = tflite::GetModel(dog_imu_model); 
  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddExpandDims(); 
  resolver.AddConv2D(); 
  resolver.AddMaxPool2D(); 
  resolver.AddFullyConnected(); 
  resolver.AddSoftmax(); 
  resolver.AddRelu(); 
  resolver.AddReshape(); 
  resolver.AddMean(); 
  
  static tflite::MicroInterpreter static_interpreter(
    tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter
  );
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  // ===== Sensor Initialization =====
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C); 
  imu.init(); 
  imu.autoOffsets(); 
  
  // MAX30102 with sample averaging for smoother signal
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
      Serial.println("[MAX30102] FAILED TO INITIALIZE");
      Serial.println("Check wiring: VIN, GND, SDA(21), SCL(22)");
      while (1) {
          delay(1000);
      }
  }
  Serial.println("[MAX30102] Initialized OK");
  
  // IMPROVED SETUP: sampleAverage = 4 for cleaner signal
  byte ledBrightness = 50;
  byte sampleAverage = 4;    // Changed from 1 to 4 for noise reduction
  byte ledMode = 2;        
  int sampleRate = 100;    
  int pulseWidth = 411;    
  int adcRange = 4096;     

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);

  sensors.begin(); 
  sensors.setWaitForConversion(false); 
  sensors.requestTemperatures(); 
  
  Serial.println("System Hardware Initialized Successfully.");
  
  digitalWrite(LED_PIN, HIGH);
  digitalWrite(BUZZER_PIN, LOW); 
  delay(150);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, HIGH); 

  xTaskCreatePinnedToCore(
    networkTaskFunc,     
    "NetworkTask",       
    12000,                
    NULL,                 
    1,                     
    &networkTaskHandle,   
    0                       
  );
}

// ============================================================
//  IMPROVED BPM UPDATE FUNCTION
// ============================================================
void updateBPM(long irValue) {

  // ----------------------------------------------------------
  // 1. Finger detection
  // ----------------------------------------------------------

  if (irValue < BPM_IR_THRESHOLD) {
    // Finger removed - clear BPM values
    realtimeBPM = 0;
    beatsPerMinute = 0;
    beatAvg = 0;

    irBaseline = 0;
    irFiltered = 0;
    irPrevious = 0;

    lastPulseTime = 0;
    lastValidPulseTime = 0;

    pulseWasRising = false;

    return;
  }

  // ----------------------------------------------------------
  // 2. Initialize baseline
  // ----------------------------------------------------------

  if (irBaseline == 0) {
    irBaseline = irValue;
    irFiltered = 0;
    irPrevious = 0;
    return;
  }

  // Slowly follow the DC level
  irBaseline =
      irBaseline * (1.0 - BASELINE_ALPHA) +
      irValue * BASELINE_ALPHA;

  // ----------------------------------------------------------
  // 3. Remove DC component
  // ----------------------------------------------------------

  float acSignal = (float)irValue - irBaseline;

  // Low-pass filter the AC signal
  irFiltered =
      irFiltered * (1.0 - SIGNAL_ALPHA) +
      acSignal * SIGNAL_ALPHA;

  // ----------------------------------------------------------
  // 4. Detect rising edge
  // ----------------------------------------------------------

  bool rising = irFiltered > irPrevious;

  // We detect a pulse when the signal starts falling
  // after reaching a sufficiently high peak.

  static float peakValue = 0;

  if (irFiltered > peakValue) {
    peakValue = irFiltered;
  }

  // Dynamic threshold based on detected peak
  float threshold = peakValue * 0.50;

  if (threshold < MIN_PULSE_AMPLITUDE) {
    threshold = MIN_PULSE_AMPLITUDE;
  }

  // Peak detected
  if (pulseWasRising &&
      !rising &&
      peakValue > threshold) {

    unsigned long now = millis();

    if (lastPulseTime > 0) {

      unsigned long interval = now - lastPulseTime;

      if (interval >= MIN_BEAT_INTERVAL &&
          interval <= MAX_BEAT_INTERVAL) {

        float bpm = 60000.0 / interval;

        if (bpm >= 40.0 && bpm <= 200.0) {

          realtimeBPM = bpm;
          beatsPerMinute = bpm;

          // Moving average
          rates[rateSpot] = (byte)bpm;
          rateSpot = (rateSpot + 1) % RATE_SIZE;

          int sum = 0;
          int count = 0;

          for (byte i = 0; i < RATE_SIZE; i++) {
            if (rates[i] > 0) {
              sum += rates[i];
              count++;
            }
          }

          if (count > 0) {
            beatAvg = sum / count;
          }

          beatDetectedCount++;

          // Debug output - shows BPM calculation
          #if DEBUG_BPM
            Serial.printf(
              "[BPM] BEAT | IR=%ld | AC=%.0f | BPM=%.1f | AVG=%d\n",
              irValue,
              irFiltered,
              realtimeBPM,
              beatAvg
            );
          #else
            Serial.printf("[BPM] %d bpm | valid\n", beatAvg);
          #endif

          lastValidPulseTime = now;
        }
      }
    }

    lastPulseTime = now;

    // Reset peak detection for next heartbeat
    peakValue = 0;
  }

  pulseWasRising = rising;
  irPrevious = irFiltered;

  // ----------------------------------------------------------
  // 5. Timeout - if no beat for 5 seconds, clear stale values
  // ----------------------------------------------------------

  if (lastValidPulseTime > 0 &&
      millis() - lastValidPulseTime > BPM_TIMEOUT_MS) {

    realtimeBPM = 0;
    beatsPerMinute = 0;
    beatAvg = 0;

    for (byte i = 0; i < RATE_SIZE; i++) {
      rates[i] = 0;
    }

    rateSpot = 0;

    lastPulseTime = 0;
    lastValidPulseTime = 0;
  }
}

// ============================================================
//  MAIN LOOP
// ============================================================
void loop() {
  unsigned long currentMillis = millis();

  // ============================================================
  // 1. BPM / HEART-RATE - USING IMPROVED ALGORITHM
  // ============================================================

  long irValue = particleSensor.getIR();
  
  // Call the improved BPM detector
  updateBPM(irValue);

  // ============================================================
  // 2. IMU, Temp, & AI Inference (20ms sync)
  // ============================================================
  if (currentMillis - lastLoopTime >= 20) {
    lastLoopTime = currentMillis;

    xyzFloat gValue = imu.getGValues();
    float filtered_X = apply_filter(0, gValue.x);
    float filtered_Y = apply_filter(1, gValue.y);
    float filtered_Z = apply_filter(2, gValue.z);
    
    if (currentMillis - lastTempReadTime >= 1000) {
      float newTemp = sensors.getTempCByIndex(0);
      if (newTemp != DEVICE_DISCONNECTED_C && newTemp > -20.0 && newTemp < 60.0) {
        tempC = newTemp;
      }
      sensors.requestTemperatures(); 
      lastTempReadTime = currentMillis;
    }
    
    sensor_buffer[buffer_index][0] = filtered_X;
    sensor_buffer[buffer_index][1] = filtered_Y;
    sensor_buffer[buffer_index][2] = filtered_Z;
    buffer_index++; 

    if (buffer_index >= TIME_STEPS) {
      for (int i = 0; i < TIME_STEPS; i++) {
        input_tensor->data.int8[i * 3 + 0] = (int8_t)(sensor_buffer[i][0] / input_tensor->params.scale + input_tensor->params.zero_point);
        input_tensor->data.int8[i * 3 + 1] = (int8_t)(sensor_buffer[i][1] / input_tensor->params.scale + input_tensor->params.zero_point);
        input_tensor->data.int8[i * 3 + 2] = (int8_t)(sensor_buffer[i][2] / input_tensor->params.scale + input_tensor->params.zero_point);
      }

      interpreter->Invoke();
      float max_prob = 0.0;
      for (int i = 0; i < 20; i++) { 
        float prob = (output_tensor->data.int8[i] - output_tensor->params.zero_point) * output_tensor->params.scale;
        if (prob > max_prob) { 
          max_prob = prob; 
          current_action_id = i; 
        }
      }

      if (WiFi.status() == WL_CONNECTED) {
        portENTER_CRITICAL(&snapshotMux);
        pendingSnapshot.sequence  = ++snapshotSequence;
        pendingSnapshot.timestamp = currentMillis;
        pendingSnapshot.bpm       = beatAvg;
        pendingSnapshot.temp      = tempC;
        pendingSnapshot.action    = current_action_id;
        pendingValid = true;
        portEXIT_CRITICAL(&snapshotMux);
        xTaskNotifyGive(networkTaskHandle);
      }
      buffer_index = 0; 
    }

    bool criticalAlert = evaluateAlerts();
    actuateHardware(criticalAlert, currentMillis);

    // Sensor logging every 1 second
    if (currentMillis - lastSensorPrintTime >= SENSOR_PRINT_INTERVAL_MS) {
      lastSensorPrintTime = currentMillis;
      logSensor(snapshotSequence, irValue, beatAvg, tempC, current_action_id, filtered_X, filtered_Y, filtered_Z);
    }

    // OLED refresh every 500ms
    if (currentMillis - lastDisplayUpdate >= DISPLAY_INTERVAL_MS) {
      lastDisplayUpdate = currentMillis;
      updateDisplay(criticalAlert);
    }
  }
}

// ============================================================
//  NETWORK TASK (Core 0)
// ============================================================
void networkTaskFunc(void *pvParameters) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    while (true) {
      portENTER_CRITICAL(&snapshotMux);
      if (!pendingValid) {
        portEXIT_CRITICAL(&snapshotMux);
        break; 
      }
      SensorSnapshot snap = pendingSnapshot;
      pendingValid = false;
      portEXIT_CRITICAL(&snapshotMux);

      if (WiFi.status() != WL_CONNECTED) {
        logNet(snap.sequence, "[WARNING] WiFi offline. Snapshot discarded.");
        break; 
      }

      networkBusy = true;
      bool ok = sendDataToCloud(snap);
      networkBusy = false;
      
      if (!ok) {
        networkState = NET_ERROR;
        logNet(snap.sequence, "[ERROR] Transmission failed. Backing off for 5s.");
        vTaskDelay(pdMS_TO_TICKS(5000));
      } else {
        networkState = NET_SUCCESS;
        // Apply 2-second rate-limiting delay to prevent server blacklisting
        vTaskDelay(pdMS_TO_TICKS(2000)); 
      }
    }
  }
}
    
// ============================================================
//  SEND DATA TO CLOUD
// ============================================================
bool sendDataToCloud(SensorSnapshot snap) {
  networkState = NET_RECORDING;
  micRecording = true;
  micHasSignal = false;
  logNet(snap.sequence, "RECORDING_AUDIO");
  
  // 8000 samples (~0.5s) to strictly eliminate memory fragmentation risks
  const int samples = 8000;

  int16_t* audio_buffer = (int16_t*)malloc(samples * sizeof(int16_t));
  if (audio_buffer == nullptr) {
      Serial.println("[MIC] ERROR: Audio buffer allocation failed");
      micRecording = false;
      return false;
  }

  int32_t raw_chunk[64];
  int samples_collected = 0;
  int32_t rawMax = 0, rawMin = 0;
  int32_t pcmMax = 0, pcmMin = 0;
  int64_t sumAbs = 0;
  bool printedRawDebug = false;

  i2s_zero_dma_buffer(I2S_PORT);
  delay(100);

  while (samples_collected < samples) {
      size_t bytes_read = 0;
      esp_err_t result = i2s_read(I2S_PORT, raw_chunk, sizeof(raw_chunk), &bytes_read, portMAX_DELAY);

      if (result != ESP_OK) {
          Serial.printf("[MIC] i2s_read ERROR = %d\n", result);
          continue;
      }
      if (bytes_read == 0) {
          Serial.println("[MIC] WARNING: ZERO BYTES FROM I2S");
          continue;
      }

      int chunk_size = bytes_read / sizeof(int32_t);

      if (!printedRawDebug) {
          Serial.printf("[MIC DEBUG] bytes_read=%u samples=%d\n", (unsigned)bytes_read, chunk_size);
          printedRawDebug = true;
      }

      for (int i = 0; i < chunk_size && samples_collected < samples; i++) {
          int32_t raw = raw_chunk[i];
          if (raw > rawMax) rawMax = raw;
          if (raw < rawMin) rawMin = raw;

          int32_t sample32 = raw >> 14;
          if (sample32 > 32767) sample32 = 32767;
          if (sample32 < -32768) sample32 = -32768;

          int16_t pcm = (int16_t)sample32;
          audio_buffer[samples_collected++] = pcm;

          if (pcm > pcmMax) pcmMax = pcm;
          if (pcm < pcmMin) pcmMin = pcm;

          sumAbs += abs((int32_t)pcm);
      }
  }

  int32_t avgAbs = sumAbs / samples;
  micMaxAbs = pcmMax;
  micAvgAbs = avgAbs;
  micHasSignal = (pcmMax != 0 || pcmMin != 0);

  if (xSemaphoreTake(serialMux, pdMS_TO_TICKS(50)) == pdTRUE) {
      Serial.println("========== MIC DIAGNOSTICS ==========");
      Serial.printf("I2S rawMin : %ld\n", (long)rawMin);
      Serial.printf("I2S rawMax : %ld\n", (long)rawMax);
      Serial.printf("PCM min    : %ld\n", (long)pcmMin);
      Serial.printf("PCM max    : %ld\n", (long)pcmMax);
      Serial.printf("PCM avgAbs : %ld\n", (long)avgAbs);
      Serial.printf("Samples    : %d\n", samples_collected);
      Serial.printf("Signal     : %s\n", micHasSignal ? "YES" : "NO");
      Serial.println("=====================================");
      xSemaphoreGive(serialMux);
  }

  // Build multipart form data
  String boundary = "RocFypBoundary123";
  String head = "--" + boundary + "\r\n";
  head += "Content-Disposition: form-data; name=\"bpm\"\r\n\r\n" + String(snap.bpm) + "\r\n";
  head += "--" + boundary + "\r\n";
  head += "Content-Disposition: form-data; name=\"temp\"\r\n\r\n" + String(snap.temp) + "\r\n";
  head += "--" + boundary + "\r\n";
  head += "Content-Disposition: form-data; name=\"action\"\r\n\r\n" + String(snap.action) + "\r\n";
  head += "--" + boundary + "\r\n";
  head += "Content-Disposition: form-data; name=\"audio_file\"; filename=\"mic_record.wav\"\r\n";
  head += "Content-Type: audio/wav\r\n\r\n"; 
  
  String tail = "\r\n--" + boundary + "--\r\n"; 
  
  size_t audioDataSize = samples * 2;
  size_t total_len = head.length() + 44 + audioDataSize + tail.length();
  
  uint8_t* payload = (uint8_t*)malloc(total_len);
  
  if (payload == NULL) {
    free(audio_buffer);
    logNet(snap.sequence, "[FATAL] Payload buffer allocation failed");
    
    // Self-healing execution when device RAM limits are breached
    Serial.println("CRITICAL: Severe memory fragmentation. Initiating self-healing reboot...");
    delay(500);
    ESP.restart(); 

    return false;
  }

  size_t offset = 0;
  memcpy(payload + offset, head.c_str(), head.length()); 
  offset += head.length();
  
  byte wavHeader[44];
  createWavHeader(wavHeader, audioDataSize);
  memcpy(payload + offset, wavHeader, 44); 
  offset += 44;
  
  memcpy(payload + offset, (uint8_t*)audio_buffer, audioDataSize); 
  offset += audioDataSize;
  
  memcpy(payload + offset, tail.c_str(), tail.length());
  
  free(audio_buffer); 

  // Print payload size (clean log)
  if (xSemaphoreTake(serialMux, pdMS_TO_TICKS(100)) == pdTRUE) {
    Serial.printf("[NET] Snapshot #%u | Payload: %u bytes\n", snap.sequence, (unsigned)total_len);
    xSemaphoreGive(serialMux);
  }

  WiFiClientSecure client;
  client.setInsecure(); 
  
  HTTPClient http;
  http.setTimeout(120000); 
  http.begin(client, serverName);
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);
  
  networkState = NET_SENDING;
  logNet(snap.sequence, "POST_STARTED");
  unsigned long postStart = millis();

  networkState = NET_WAITING;
  int httpResponseCode = http.POST(payload, total_len);

  unsigned long latency = millis() - postStart;
  bool success = false;

  if (httpResponseCode == 200) {
    String responseStr = http.getString();
    parseServerResponse(responseStr, snap.sequence, micMaxAbs, micHasSignal, httpResponseCode, latency, total_len);
    success = true;
  } else if (httpResponseCode > 0) {
    if (xSemaphoreTake(serialMux, pdMS_TO_TICKS(50)) == pdTRUE) {
      Serial.printf("[NET] #%u | HTTP_ERROR %d | latency=%lums\n", snap.sequence, httpResponseCode, latency);
      xSemaphoreGive(serialMux);
    }
  } else {
    if (xSemaphoreTake(serialMux, pdMS_TO_TICKS(50)) == pdTRUE) {
      Serial.printf("[NET] #%u | REQUEST_FAILED: %s | latency=%lums\n",
                    snap.sequence, http.errorToString(httpResponseCode).c_str(), latency);
      xSemaphoreGive(serialMux);
    }
  }

  http.end();
  client.stop(); // Free WiFi resources
  free(payload); 
  return success;
}

// ============================================================
//  PARSE SERVER RESPONSE - NEW PROFESSIONAL FORMAT
// ============================================================
void parseServerResponse(
  String responseStr,
  uint32_t sequence,
  int32_t micMax,
  bool micSignal,
  int httpStatus,
  unsigned long latency,
  size_t payloadSize
) {
  StaticJsonDocument<2048> doc;
  DeserializationError error = deserializeJson(doc, responseStr);
  
  if (xSemaphoreTake(serialMux, pdMS_TO_TICKS(100)) != pdTRUE) {
    return;
  }

  Serial.println();
  Serial.println("========================================");
  Serial.println("       CLOUD MULTIMODAL FUSION");
  Serial.println("========================================");

  Serial.printf("Snapshot       : #%u\n", sequence);

  if (!error) {
    // Status
    const char* status = doc["status"] | "UNKNOWN";
    Serial.printf("Status         : %s\n", status);
    Serial.println();

    // Physiological
    Serial.println("Physiological");
    Serial.println("----------------------------------------");
    float serverTemp = doc["inputs_received"]["temp"] | 0.0;
    int serverBpm = doc["inputs_received"]["bpm"] | 0;
    int serverAction = doc["inputs_received"]["action"] | 0;
    Serial.printf("Temperature    : %.2f °C\n", serverTemp);
    Serial.printf("Heart Rate     : %d bpm\n", serverBpm);
    Serial.printf("Motion Class   : %d\n", serverAction);
    Serial.println();

    // Audio
    Serial.println("Audio");
    Serial.println("----------------------------------------");
    float peakVolume = doc["audio_diagnostics"]["peak_volume"] | 0.0;
    float audioRisk = doc["audio_diagnostics"]["final_audio_risk"] | 0.0;
    Serial.printf("Mic Level      : %ld\n", (long)micMax);
    Serial.printf("Signal         : %s\n", micSignal ? "DETECTED" : "LOW / SILENT");
    Serial.printf("Peak Volume    : %.2f\n", peakVolume);
    Serial.printf("Audio Risk     : %.2f\n", audioRisk);
    Serial.println();

    // Fusion
    Serial.println("Fusion");
    Serial.println("----------------------------------------");
    int emotionClass = doc["emotion_class"] | -1;
    const char* emotion = doc["diagnosed_emotion"] | "UNKNOWN";
    Serial.printf("Emotion Class  : %d\n", emotionClass);
    Serial.printf("Emotion        : %s\n", emotion);
    Serial.println("Fusion Status  : SUCCESS");
    Serial.println();

    // Network
    Serial.println("Network");
    Serial.println("----------------------------------------");
    Serial.printf("HTTP Status    : %d\n", httpStatus);
    Serial.printf("Latency        : %lu ms\n", latency);
    Serial.printf("Payload        : %u bytes\n", (unsigned)payloadSize);
  } else {
    Serial.println("Status         : JSON PARSE ERROR");
    Serial.println();
    Serial.println("Raw Server Response:");
    Serial.println(responseStr);
  }

  Serial.println("========================================");
  Serial.println();

  xSemaphoreGive(serialMux);
}

// ============================================================
//  EVALUATE ALERTS
// ============================================================
bool evaluateAlerts() {
  bool criticalAlert = false;
  
  #ifdef DEMO_MODE
    if (particleSensor.getIR() > 50000 && beatAvg == 0) beatAvg = 85; 
    if (tempC > 31.0) criticalAlert = true; 
    if (current_action_id > 2 && beatAvg > 60) criticalAlert = true; 
  #else
    // Actual production diagnostic thresholds
    if (tempC > 39.5) criticalAlert = true; 
    if (current_action_id > 5 && beatAvg > 150) criticalAlert = true; 
  #endif
  
  return criticalAlert;
}

// ============================================================
//  ACTUATE HARDWARE
// ============================================================
void actuateHardware(bool criticalAlert, unsigned long currentMillis) {
  static unsigned long lastBlinkTime = 0;
  static bool alertToggle = false;

  if (criticalAlert) {
    if (currentMillis - lastBlinkTime > 200) {
      alertToggle = !alertToggle; 
      digitalWrite(LED_PIN, alertToggle ? HIGH : LOW);     
      digitalWrite(BUZZER_PIN, alertToggle ? LOW : HIGH); 
      lastBlinkTime = currentMillis;
    }
  } else {
    digitalWrite(LED_PIN, LOW);     
    digitalWrite(BUZZER_PIN, HIGH); 
    alertToggle = false;
  }
}

// ============================================================
//  UPDATE OLED DISPLAY
// ============================================================
void updateDisplay(bool criticalAlert) {
  display.clearDisplay(); 
  display.setCursor(0,0);
  
  if (criticalAlert) {
    display.setTextColor(BLACK, WHITE);
    display.println(" !! CRITICAL ALERT !! "); 
  } else {
    display.setTextColor(WHITE, BLACK);
    display.println(" STATUS: NORMAL "); 
  }
  
  display.setTextColor(WHITE, BLACK);
  display.println("---------------------"); 
  
  display.print("BPM   : "); 
  display.println(beatAvg); 
  display.print("Temp  : "); 
  display.print(tempC, 1); 
  display.println(" C"); 
  display.print("Action: Class "); 
  display.println(current_action_id); 
  
  display.print("Mic   : ");
  if (micRecording) {
    display.println("REC");
  } else if (micHasSignal) {
    display.println("OK");
  } else {
    display.println("LOW");
  }
  
  if(WiFi.status() == WL_CONNECTED) {
    display.println("WiFi  : CONNECTED");
  } else {
    display.println("WiFi  : OFFLINE");
  }
  
  display.display();
}