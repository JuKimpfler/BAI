// src/main.cpp
#include <Arduino.h>
#include "roboter_lut_actions.h"

static inline int clamp_i(int v, int lo, int hi) {
  return (v < lo) ? lo : (v > hi) ? hi : v;
}

static inline uint8_t lut_get_action(float distance_cm, float rel_angle_deg) {
  // rel_angle_deg kann [-180..180] sein. Wir mappen auf [0..360)
  float a = rel_angle_deg;
  while (a < 0) a += 360.0f;
  while (a >= 360.0f) a -= 360.0f;

  int angle_idx = (int)lroundf(a / (float)LUT_ANGLE_STEP_DEG) % LUT_ANGLE_COUNT;
  int dist_idx  = clamp_i((int)lroundf(distance_cm / (float)LUT_DIST_STEP_CM), 0, LUT_DIST_COUNT - 1);

  uint32_t idx = (uint32_t)dist_idx * (uint32_t)LUT_ANGLE_COUNT + (uint32_t)angle_idx;
  return LUT_ACTIONS[idx];
}

static inline float action_to_rel_angle_deg(uint8_t action) {
  const float step = 360.0f / 90.0f;   // 4°
  float ang = action * step;           // 0..356
  if (ang > 180.0f) ang -= 360.0f;     // -> [-180..180]
  return ang;
}

void setup() {
  Serial.begin(115200);
  Serial.println("LUT demo");
}

void loop() {
  float dist_cm = 120;
  float rel_ball_deg = -45; // dein gemessener relativer Winkel zum Ball

  uint8_t action = lut_get_action(dist_cm, rel_ball_deg);
  float drive_rel_deg = action_to_rel_angle_deg(action);

  Serial.print("action="); Serial.print(action);
  Serial.print(" -> drive_rel_deg="); Serial.println(drive_rel_deg);

  delay(500);
}
