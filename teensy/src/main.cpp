/**
 * main.cpp – Teensy 4.0 Roboter-Steuerung mit LUT
 * =================================================
 * Liest Winkel und Abstand vom Sensor, schlägt die optimale Fahrtrichtung
 * in der vorberechneten Lookup-Tabelle (robot_lut.h) nach und gibt die
 * Motorsteuerung aus.
 *
 * Hardware-Voraussetzungen (Beispiel-Verdrahtung, bitte anpassen):
 *   - Entfernungssensor (z. B. HC-SR04 oder VL53L0X) → Pin TRIG/ECHO
 *   - Kompass / IMU (z. B. BNO055 via I²C) → liefert absoluten Winkel
 *   - Motortreiber (z. B. L298N oder DRV8833)
 *     · Motor A: ENA, IN1, IN2
 *     · Motor B: ENB, IN3, IN4
 *
 * Kompilieren & Flashen:
 *   pio run --target upload
 */

#include <Arduino.h>
#include "robot_lut.h"   // <── generiert mit generate_lut.py

// ─── Hardware-Pins (bitte auf eigene Verdrahtung anpassen) ───────────────────
// Ultraschall-Sensor HC-SR04
static constexpr uint8_t PIN_TRIG = 6;
static constexpr uint8_t PIN_ECHO = 7;

// Kompass / Winkel-Eingang (analoges Beispiel; bei digitalem Sensor anpassen)
// Hier wird ein einfaches Poti-Beispiel gezeigt – im echten Einsatz
// durch IMU-Bibliothek ersetzen (z. B. Adafruit_BNO055).
static constexpr uint8_t PIN_WINKEL_ANALOG = A0;

// Motortreiber (z. B. L298N)
static constexpr uint8_t PIN_ENA = 2;
static constexpr uint8_t PIN_IN1 = 3;
static constexpr uint8_t PIN_IN2 = 4;
static constexpr uint8_t PIN_ENB = 5;
static constexpr uint8_t PIN_IN3 = 8;
static constexpr uint8_t PIN_IN4 = 9;

// PWM-Grundgeschwindigkeit (0–255)
static constexpr uint8_t BASIS_SPEED = 180;

// ─── Sensor-Hilfsfunktionen ──────────────────────────────────────────────────

/**
 * Misst den Abstand in cm mit einem HC-SR04 Ultraschall-Sensor.
 * Gibt 0 zurück, wenn kein Echo empfangen wurde.
 */
int messeAbstandCm() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);

  long dauer = pulseIn(PIN_ECHO, HIGH, 25000UL);  // Timeout 25 ms
  if (dauer == 0) return LUT_ABSTAND_MAX;          // kein Echo → Maximalwert

  int abstand = (int)(dauer * 0.01715f);           // cm = µs × (34300 cm/s / 2 / 1e6)
  if (abstand < LUT_ABSTAND_MIN) abstand = LUT_ABSTAND_MIN;
  if (abstand > LUT_ABSTAND_MAX) abstand = LUT_ABSTAND_MAX;
  return abstand;
}

/**
 * Liest den relativen Winkel zum Ziel (0–359°).
 *
 * Im echten Einsatz: Winkel von IMU/Kompass lesen und mit der Zielrichtung
 * verrechnen.  Diese Demo liest ein Analogpoti (A0) als Platzhalter.
 */
int leseWinkelDeg() {
  int rohwert  = analogRead(PIN_WINKEL_ANALOG);    // 0–1023
  int winkelDeg = (int)((rohwert / 1023.0f) * 359.0f);
  return winkelDeg;
}

// ─── Motor-Hilfsfunktionen ───────────────────────────────────────────────────

/** Fährt vorwärts. */
void vorwaerts() {
  analogWrite(PIN_ENA, BASIS_SPEED);
  analogWrite(PIN_ENB, BASIS_SPEED);
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, HIGH);
  digitalWrite(PIN_IN4, LOW);
}

/** Dreht nach links (auf der Stelle). */
void dreheLinks() {
  analogWrite(PIN_ENA, BASIS_SPEED);
  analogWrite(PIN_ENB, BASIS_SPEED);
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, HIGH);
  digitalWrite(PIN_IN3, HIGH);
  digitalWrite(PIN_IN4, LOW);
}

/** Dreht nach rechts (auf der Stelle). */
void dreheRechts() {
  analogWrite(PIN_ENA, BASIS_SPEED);
  analogWrite(PIN_ENB, BASIS_SPEED);
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, HIGH);
}

/** Hält an. */
void stop() {
  analogWrite(PIN_ENA, 0);
  analogWrite(PIN_ENB, 0);
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, LOW);
}

/**
 * Führt den aus der LUT ermittelten Aktionsindex aus.
 *
 * Konvention (passend zum Training):
 *   Aktion  0              → Vorwärts (0°)
 *   Aktion 1–44  (1–176°)  → Links drehen
 *   Aktion 45              → Rückwärts (180°)
 *   Aktion 46–89 (184–356°)→ Rechts drehen (=kleiner Linkswinkel von der anderen Seite)
 *
 * Im echten Fahrzeug sollte hier ein PID-Regler oder eine Kurvenfahrt
 * implementiert werden.  Diese Demo zeigt nur Vorwärts / Links / Rechts.
 */
void fuehreAktionAus(uint8_t aktion) {
  float grad = robot_aktion_zu_winkel(aktion);

  if (grad < 10.0f || grad > 350.0f) {
    // Fast geradeaus → vorwärts
    vorwaerts();
  } else if (grad <= 180.0f) {
    // Linkskurve
    dreheLinks();
  } else {
    // Rechtskurve
    dreheRechts();
  }
}

// ─── Setup ───────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}  // Warte max 3 s auf seriellen Monitor

  Serial.println(F("Roboter LUT-Steuerung startet..."));
  Serial.print(F("LUT-Eintraege: "));
  Serial.println(LUT_GESAMT);
  Serial.print(F("Aktionen     : 0–"));
  Serial.println(LUT_AKTIONEN - 1);
  Serial.print(F("Winkel/Aktion: "));
  Serial.print(LUT_WINKEL_SCHRITT);
  Serial.println(F("°"));

  // Pins konfigurieren
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_WINKEL_ANALOG, INPUT);

  pinMode(PIN_ENA, OUTPUT);
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_ENB, OUTPUT);
  pinMode(PIN_IN3, OUTPUT);
  pinMode(PIN_IN4, OUTPUT);

  stop();
  Serial.println(F("Bereit."));
}

// ─── Hauptschleife ────────────────────────────────────────────────────────────
void loop() {
  // 1. Sensordaten lesen
  int abstandCm = messeAbstandCm();
  int winkelDeg = leseWinkelDeg();

  // 2. LUT-Lookup  (O(1), kein Netzwerk zur Laufzeit nötig!)
  uint8_t aktion = robot_lut_lookup(winkelDeg, abstandCm);
  float   grad   = robot_aktion_zu_winkel(aktion);

  // 3. Debug-Ausgabe
  Serial.print(F("Winkel="));   Serial.print(winkelDeg);
  Serial.print(F("°  Abst="));  Serial.print(abstandCm);
  Serial.print(F("cm  Akt="));  Serial.print(aktion);
  Serial.print(F("  →"));       Serial.print(grad, 1);
  Serial.println(F("°"));

  // 4. Motor-Steuerung ausführen
  if (abstandCm <= 5) {
    // Zu nah → anhalten
    stop();
    Serial.println(F("STOP – Ziel erreicht oder Hindernis!"));
  } else {
    fuehreAktionAus(aktion);
  }

  delay(100);  // 10 Hz Regelschleife
}
