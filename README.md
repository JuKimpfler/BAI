# Projektübersicht: KI-gesteuerte Omnidirektionale Roboter-Navigation

1.218 Params

## 🎯 1. Das Projektziel
Entwicklung eines intelligenten Steuerungssystems für einen omnidirektionalen Roboter. Der Roboter soll ein Objekt in seiner Umgebung anvisieren und eine optimale, kollisionsfreie Route berechnen, um sich direkt vor dem Objekt zu positionieren (Winkel 0°, Abstand 0 cm). Die Entscheidungsfindung (Kognition) wird von einem trainierten Neuronalen Netz (Small/Micro AI) übernommen.

## ⚙️ 2. System-Architektur (Ziel-Zustand)
Das System wird im "Master-Slave" oder "Co-Prozessor" Prinzip aufgebaut, um die Echtzeitfähigkeit des Roboters zu garantieren.

*   **Der Muskel (Teensy 3.x/4.x):**
    *   Läuft mit 500 - 1000 Hz.
    *   Zuständig für zeitkritische Aufgaben: Sensoren auslesen, Motor-PID-Regler, Kinematik des Omnidrives (Umrechnung von X/Y/Rotation in einzelne Radgeschwindigkeiten).
    *   Sendet via UART (Seriell) kontinuierlich: `Aktueller_Winkel, Aktueller_Abstand`.
*   **Das Gehirn (Raspberry Pi Zero / Zero W):**
    *   Empfängt Sensordaten via UART.
    *   Führt das trainierte KI-Modell aus (Inference).
    *   Sendet via UART zurück: `Fahrvektor_X, Fahrvektor_Y` (Werte zwischen -1.0 und 1.0).
*   **Kommunikation:** Seriell (UART) mit hoher Baudrate (z.B. 1.000.000 Baud), asynchron (non-blocking), damit der Teensy niemals auf den Pi warten muss.

---

## 🗺️ 3. Der Fahrplan (Projekt-Phasen)

### Phase 1: Daten-Generierung & Simulation (Rein Software) 📍 *Hier starten wir*
Da KI-Modelle aus Beispielen lernen (Supervised Learning), müssen wir dem Modell erst zeigen, wie die perfekten Routen aussehen.
*   **Aufgabe:** Ein Python-Skript schreiben, das 10.000 bis 50.000 zufällige Positionen (Winkel & Abstand) generiert.
*   **Logik:** Mathematische Berechnung der perfekten Fahrvektoren (Ausweichbogen fahren, wenn das Objekt hinten/seitlich ist; geradeaus fahren, wenn es vorne ist).
*   **Ergebnis:** Eine CSV-Datei (`roboter_trainingsdaten.csv`), die als "Lehrbuch" für die KI dient.

### Phase 2: KI-Modell bauen & Trainieren (Rein Software)
Jetzt wird das Gehirn gebaut und mit der CSV-Datei aus Phase 1 unterrichtet.
*   **Werkzeuge:** Python, PyTorch, Pandas (für CSV).
*   **Aufbau:** Ein kleines Feed-Forward-Netzwerk (z.B. 2 Inputs $\rightarrow$ 32 Neuronen $\rightarrow$ 32 Neuronen $\rightarrow$ 2 Outputs).
*   **Training:** Das Netzwerk liest die Daten, rät die Vektoren, vergleicht sie mit den perfekten simulierten Vektoren und korrigiert seine Gewichte.
*   **Visualisierung:** Wir schreiben ein kleines Skript (z.B. mit `matplotlib`), das uns auf dem PC virtuell aufzeichnet, wie die frisch trainierte KI fahren würde. So testen wir, ob sie es verstanden hat, ohne Hardware zu riskieren.
*   **Ergebnis:** Eine Datei (z.B. `modell_gewichte.pth`), die das trainierte "Wissen" enthält.

### Phase 3: Vorbereitung für den Pi Zero (Optimierung)
Ein normales PyTorch-Modell ist für den alten ARM-Prozessor des Pi Zero oft etwas zu "schwerfällig" (braucht lange zum Laden, frisst RAM).
*   **Aufgabe:** Das fertige Modell aus Phase 2 in ein leichtgewichtigeres Format konvertieren.
*   **Werkzeuge:** ONNX (Open Neural Network Exchange) oder TensorFlow Lite (TFLite).
*   **Test:** Ein Test-Skript auf dem PC, das prüft, ob das verkleinerte Modell noch exakt die gleichen Vektoren ausspuckt.

### Phase 4: Hardware-Integration & Test
Der Sprung von der Simulation in die Realität ("Sim2Real").
*   **Raspberry Pi:** Python-Skript installieren, das den UART-Port abhört und das TFLite-Modell füttert.
*   **Teensy:** C++ Code anpassen, sodass er Dummy-Daten (oder erste echte Sensordaten) an den Pi sendet und die Motor-Vektoren ausliest.
*   **Trockentest:** Den Roboter aufbocken (Räder in der Luft) und schauen, ob die Motoren richtig auf simulierte Objekt-Positionen reagieren.
*   **Feinschliff:** In der Realität gibt es Radschlupf und Sensor-Rauschen. Eventuell muss das Modell mit künstlichem "Rauschen" in Phase 1 noch robuster trainiert werden.

---

## 🛠️ 4. Deine nächsten Schritte
Wir sind in **Phase 1 und 2**. Du brauchst aktuell noch keinen Pi Zero und keinen Teensy anzuschließen. 

**Möchtest du, dass wir als nächstes den Code für das PyTorch-Netzwerk (Phase 2) aufbauen, welches die in der letzten Nachricht simulierten Daten einliest und lernt?**