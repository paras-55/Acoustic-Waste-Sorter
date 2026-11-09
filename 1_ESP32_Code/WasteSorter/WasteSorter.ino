/*
  WasteSorter.ino  —  ONE firmware for the whole prototype
  Board: ESP32 Dev Module (ESP32 DevKit V1)   Core: esp32 by Espressif 3.x
  Library: ESP32Servo (Kevin Harrington)

  What it does
  ------------
  * INMP441 microphone over I2S (ESP-IDF "i2s_std" driver, used directly so
    every setting is explicit — no hidden mono/slot defaults).
  * Reads BOTH 32-bit I2S slots and finds out at boot which slot actually
    carries the microphone (diagnostic 'D'). This is the fix for the
    "99.8 % exact zeros" problem: reading the slot where the INMP441 is
    silent (tri-stated) gives zeros with rare glitch spikes.
  * Triggered capture: waits for an impact, keeps 0.2 s before it and
    1.3 s after it (1.5 s clip, 16 kHz, signed 16-bit mono).
  * MG996R servo on GPIO 18 tilts the impact tray / flap.

  Serial protocol (921600 baud, used by the Python scripts)
  ---------------------------------------------------------
  Laptop -> ESP32 (single characters):
    D        run microphone diagnostic, prints DIAG lines
    R        record a clip immediately (no trigger)
    T        arm trigger, record when an impact happens (timeout 15 s)
    P / M    sort: tilt to plastic / metal side, hold, return to centre
    C        centre the servo
    A<deg>   move servo to an angle, e.g. A60   (for calibrating angles)
    G<n>     set gain shift 12..18, e.g. G15  (smaller n = louder)
    ?        print STATUS line
  ESP32 -> laptop:
    Text lines ending in '\n' (READY, DIAG ..., STATUS ..., PLASTIC_OK ...)
    Audio is sent ONLY as a framed binary block:
      A5 5A 'W' 'A' 'V' | status(1) | slot(1) | nSamples(uint32 LE)
      | nSamples * int16 LE | byteSum(uint32 LE)
      status: 0 = immediate clip, 1 = triggered clip, 2 = trigger timeout (n=0)
*/

#include <Arduino.h>
#include <ESP32Servo.h>
#include <algorithm>
#include "driver/i2s_std.h"

// ---------------- Pins (match your wiring) ----------------
static const int        PIN_SERVO = 18;
static const gpio_num_t PIN_BCLK  = GPIO_NUM_26;   // INMP441 SCK
static const gpio_num_t PIN_WS    = GPIO_NUM_25;   // INMP441 WS
static const gpio_num_t PIN_SD    = GPIO_NUM_33;   // INMP441 SD

// ---------------- Audio settings ----------------
static const uint32_t SERIAL_BAUD   = 921600;
static const uint32_t SAMPLE_RATE   = 16000;
static const int      CLIP_SAMPLES  = 24000;   // 1.5 s
static const int      PRE_SAMPLES   = 3200;    // 0.2 s kept before the impact
static const int      WARMUP_SAMPLES = 4800;   // 0.3 s settle + noise measurement
static const int      TRIG_MIN      = 300;     // minimum trigger level (int16 units)
static const int      TRIG_K        = 10;      // trigger = K x average noise level
static const uint32_t TRIG_TIMEOUT_MS = 15000;
static int            gainShift     = 17;      // 32-bit I2S word >> gainShift -> int16

// ---------------- Servo settings ----------------
static const int ANGLE_CENTER  = 90;
static const int ANGLE_PLASTIC = 45;
static const int ANGLE_METAL   = 135;
static const int SORT_HOLD_MS  = 900;          // how long the tray stays tilted

Servo diverter;
i2s_chan_handle_t rxChan = NULL;
int     activeSlot = 0;       // which 32-bit word of each stereo frame is the mic
bool    micOk = false;
int16_t *clip = NULL;
static int32_t frameBuf[512]; // 256 stereo frames

// ------------------------------------------------------------------
bool i2sInit() {
  i2s_chan_config_t chanCfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  chanCfg.dma_desc_num  = 8;
  chanCfg.dma_frame_num = 256;
  if (i2s_new_channel(&chanCfg, NULL, &rxChan) != ESP_OK) return false;

  i2s_std_config_t cfg = {
    .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
    .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
    .gpio_cfg = {
      .mclk = I2S_GPIO_UNUSED,
      .bclk = PIN_BCLK,
      .ws   = PIN_WS,
      .dout = I2S_GPIO_UNUSED,
      .din  = PIN_SD,
      .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
    },
  };
  if (i2s_channel_init_std_mode(rxChan, &cfg) != ESP_OK) return false;
  if (i2s_channel_enable(rxChan) != ESP_OK) return false;
  return true;
}

// Returns number of complete stereo frames read into frameBuf.
size_t readFrames() {
  size_t gotBytes = 0;
  if (i2s_channel_read(rxChan, frameBuf, sizeof(frameBuf), &gotBytes, 1000) != ESP_OK) return 0;
  return gotBytes / (2 * sizeof(int32_t));
}

void discardFrames(int frames) {
  int done = 0;
  while (done < frames) {
    size_t n = readFrames();
    if (n == 0) return;
    done += n;
  }
}

// ------------------------------------------------------------------
// Diagnostic: 1 s of raw data, statistics for each slot separately.
// A working INMP441 slot is almost never exactly zero (its noise floor
// toggles the low bits continuously). The silent slot is ~100 % zeros.
void runDiagnostic() {
  struct S { uint32_t n = 0, nz = 0; double sum = 0, sumsq = 0; int32_t peak = 0; uint32_t bits = 0; int32_t prev = 0; } st[2];

  discardFrames(3200);                     // flush stale DMA buffers
  while (st[0].n < SAMPLE_RATE) {
    size_t n = readFrames();
    if (n == 0) { Serial.println("DIAG ERROR i2s_read_timeout (no BCLK/WS? check SCK=26, WS=25)"); micOk = false; return; }
    for (size_t i = 0; i < n; i++) {
      for (int s = 0; s < 2; s++) {
        int32_t w = frameBuf[2 * i + s];
        int32_t v = w >> 8;                // INMP441: 24-bit data, MSB-aligned in 32-bit slot
        S &x = st[s];
        x.n++;
        if (w != 0) x.nz++;
        x.sum += v; x.sumsq += (double)v * v;
        if (abs(v) > x.peak) x.peak = abs(v);
        x.bits |= (uint32_t)(w ^ x.prev);  // which raw bits ever change
        x.prev = w;
      }
    }
  }

  int best = 0;
  float frac[2]; double rms[2];
  for (int s = 0; s < 2; s++) {
    S &x = st[s];
    double mean = x.sum / x.n;
    double var = x.sumsq / x.n - mean * mean;
    rms[s] = var > 0 ? sqrt(var) : 0;
    frac[s] = (float)x.nz / x.n;
    Serial.printf("DIAG slot%d nonzero=%u/%u (%.1f%%) mean24=%.1f rms24=%.1f peak24=%d togglingBits=0x%08X\n",
                  s, x.nz, x.n, 100.0f * frac[s], mean, rms[s], x.peak, x.bits);
  }
  if (frac[1] > frac[0] || (frac[1] == frac[0] && rms[1] > rms[0])) best = 1;

  if (frac[best] > 0.90f && rms[best] > 1.0) {
    activeSlot = best; micOk = true;
    Serial.printf("DIAG RESULT MIC_OK slot=%d\n", best);
  } else {
    micOk = false;
    Serial.printf("DIAG RESULT MIC_FAULT best_slot=%d nonzero=%.1f%%  (both slots mostly zero: check SD=33 wire, 3V3 on VDD, L/R tied to GND, try another breadboard row)\n",
                  best, 100.0f * frac[best]);
  }
}

// ------------------------------------------------------------------
void sendFrame(uint8_t status, const int16_t *data, uint32_t n) {
  const uint8_t magic[5] = {0xA5, 0x5A, 'W', 'A', 'V'};
  uint32_t sum = 0;
  const uint8_t *p = (const uint8_t *)data;
  for (uint32_t i = 0; i < n * 2; i++) sum += p[i];
  Serial.write(magic, 5);
  Serial.write(status);
  Serial.write((uint8_t)activeSlot);
  Serial.write((const uint8_t *)&n, 4);        // ESP32 is little-endian
  if (n) Serial.write(p, n * 2);
  Serial.write((const uint8_t *)&sum, 4);
  Serial.flush();
}

// Fills clip[] in time order. Returns status 0/1 on success, 2 on timeout.
uint8_t captureClip(bool triggered) {
  float dc = 0.0f, noiseAcc = 0.0f;
  int   noiseN = 0, warm = 0, head = 0, filled = 0;
  long  postLeft = -1;
  int   thr = TRIG_MIN;
  bool  done = false;
  uint32_t t0 = millis();

  discardFrames(1600);                     // flush stale DMA buffers (0.1 s)
  while (!done) {
    if (triggered && millis() - t0 > TRIG_TIMEOUT_MS) return 2;
    size_t n = readFrames();
    if (n == 0) return 2;
    for (size_t i = 0; i < n && !done; i++) {
      int32_t w = frameBuf[2 * i + activeSlot];
      float x = (float)(w >> gainShift);
      dc += 0.001f * (x - dc);             // remove the mic's DC offset
      int32_t q = lrintf(x - dc);
      if (q > 32767) q = 32767;
      if (q < -32768) q = -32768;

      if (warm < WARMUP_SAMPLES) {         // settle DC filter, measure noise
        warm++;
        if (warm > WARMUP_SAMPLES / 3) { noiseAcc += fabsf(x - dc); noiseN++; }
        if (warm == WARMUP_SAMPLES) {
          int nt = (int)(TRIG_K * (noiseAcc / max(noiseN, 1)));
          thr = max(TRIG_MIN, nt);
          if (triggered) Serial.printf("INFO armed threshold=%d noise=%.1f -> drop the object now\n", thr, noiseAcc / max(noiseN, 1));
        }
        continue;
      }

      clip[head] = (int16_t)q;
      head = (head + 1) % CLIP_SAMPLES;
      filled++;

      if (!triggered) {
        if (filled >= CLIP_SAMPLES) done = true;
      } else if (postLeft < 0) {
        if (filled >= PRE_SAMPLES && abs(q) >= thr) postLeft = CLIP_SAMPLES - PRE_SAMPLES - 1;
      } else {
        if (postLeft == 0) done = true; else postLeft--;
      }
    }
  }
  std::rotate(clip, clip + head, clip + CLIP_SAMPLES);   // oldest sample first
  return triggered ? 1 : 0;
}

// ------------------------------------------------------------------
void sortTo(int angle, const char *ack) {
  diverter.write(angle);
  delay(SORT_HOLD_MS);
  diverter.write(ANGLE_CENTER);
  delay(400);
  Serial.println(ack);
}

void printStatus() {
  Serial.printf("STATUS mic=%s slot=%d gain_shift=%d rate=%u clip=%d pre=%d\n",
                micOk ? "OK" : "FAULT", activeSlot, gainShift, SAMPLE_RATE, CLIP_SAMPLES, PRE_SAMPLES);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(300);

  diverter.setPeriodHertz(50);
  diverter.attach(PIN_SERVO, 500, 2500);
  diverter.write(ANGLE_CENTER);

  clip = (int16_t *)malloc(CLIP_SAMPLES * sizeof(int16_t));
  if (!clip) { Serial.println("ERROR out_of_memory"); }

  if (!i2sInit()) {
    Serial.println("ERROR i2s_init_failed");
  } else {
    delay(200);
    runDiagnostic();
  }
  printStatus();
  Serial.println("READY");
}

void loop() {
  if (!Serial.available()) return;
  char c = Serial.read();
  switch (c) {
    case 'D': case 'd': runDiagnostic(); printStatus(); break;
    case 'R': case 'r':
    case 'T': case 't': {
      bool trig = (c == 'T' || c == 't');
      if (!micOk) Serial.println("WARN mic_fault_recording_anyway");
            diverter.detach();                   // silence servo hum/PWM while listening
      uint8_t st = captureClip(trig);
      diverter.attach(PIN_SERVO, 500, 2500);
      diverter.write(ANGLE_CENTER);
      if (st == 2) sendFrame(2, clip, 0);
      else sendFrame(st, clip, CLIP_SAMPLES);
      break;
    }
    case 'P': case 'p': sortTo(ANGLE_PLASTIC, "PLASTIC_OK"); break;
    case 'M': case 'm': sortTo(ANGLE_METAL, "METAL_OK"); break;
    case 'C': case 'c': diverter.write(ANGLE_CENTER); Serial.println("CENTER_OK"); break;
    case 'A': case 'a': {
      int a = Serial.parseInt();
      a = constrain(a, 0, 180);
      diverter.write(a);
      Serial.printf("ANGLE_OK %d\n", a);
      break;
    }
    case 'G': case 'g': {
      int g = Serial.parseInt();
      gainShift = constrain(g, 12, 18);
      Serial.printf("GAIN_OK %d\n", gainShift);
      break;
    }
    case '?': printStatus(); break;
    default: break;                        // ignore \r, \n and unknown bytes
  }
}
