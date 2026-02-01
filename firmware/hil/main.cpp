#include <Arduino.h>
#include <stdlib.h>
#include <string.h>

// TODO: include your TFLite Micro headers and model data here.
// #include "model_data.h"
// #include "tensorflow/lite/micro/all_ops_resolver.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/schema/schema_generated.h"

// Simple placeholder inference that maps ay to a vibration intensity.
// Replace with real TFLite Micro interpreter invocation.
float run_inference(float ax, float ay, float az, float gx, float gy, float gz) {
  float mag = fabs(ay) + 0.1f * fabs(gx) + 0.1f * fabs(gy);
  float vib = mag / 20.0f;  // scale heuristic
  if (vib > 1.0f) vib = 1.0f;
  if (vib < 0.0f) vib = 0.0f;
  return vib;
}

static const uint16_t SERIAL_BAUD = 115200;
static const uint16_t LINE_MAX = 160;

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial) {
    ;
  }
}

bool parse_line(char *line, float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  // Expected: T,ax,ay,az,gx,gy,gz
  if (line[0] != 'T') return false;
  char *ctx = nullptr;
  // Skip the leading token
  strtok_r(line, ",", &ctx);
  char *tok = nullptr;
  float vals[6];
  for (int i = 0; i < 6; ++i) {
    tok = strtok_r(nullptr, ",", &ctx);
    if (!tok) return false;
    vals[i] = atof(tok);
  }
  ax = vals[0];
  ay = vals[1];
  az = vals[2];
  gx = vals[3];
  gy = vals[4];
  gz = vals[5];
  return true;
}

void loop() {
  static char buffer[LINE_MAX];
  static uint16_t idx = 0;

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || idx >= LINE_MAX - 1) {
      buffer[idx] = '\0';
      idx = 0;

      float ax, ay, az, gx, gy, gz;
      if (parse_line(buffer, ax, ay, az, gx, gy, gz)) {
        float vib = run_inference(ax, ay, az, gx, gy, gz);
        Serial.print("A,");
        Serial.println(vib, 4);
      }
    } else if (c != '\r') {
      buffer[idx++] = c;
    }
  }
}
