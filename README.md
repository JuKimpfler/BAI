# BAI Studio

Dieses Repository ist auf das zentrale Desktop-Tool **`BAI_Studio.py`** ausgerichtet.  
Das Studio bündelt Training, LUT-Erstellung, LUT-Glättung und Analyse in einer einzigen PyQt5-Oberfläche.

## Überblick

`BAI_Studio.py` stellt drei Hauptbereiche bereit:

- **Workflow**  
  Konfigurierbare Pipeline mit aktivierbaren Schritten:
  1. Training
  2. LUT erstellen
  3. LUT glätten
  4. Analyser
  5. LUT Analyser

- **Tools**  
  Startet die bestehenden Einzel-GUIs direkt aus dem Studio.

- **Ergebnisse**  
  Zeigt Workflow-Zusammenfassung und Heatmaps (Model/LUT-Erfolgsrate).

## Wichtige Workflow-Parameter (aus der aktuellen Studio-Datei)

- **Training:** Legt Lernaufwand und Modellgröße fest (Epochen, Neuronen, Batch-Größe, CPU-Threads, Modelldatei).
- **Epsilon-Schedule:** Definiert den Explorationsverlauf im DQN-Training als **newline-separierte** Einträge `Trainingsfortschritt in %:Epsilon-Wert`, z. B.:
  ```text
  0:1.0
  50:0.5
  100:0.02
  ```
- **Lookup Table (LUT):** Steuert Ein-/Ausgabedateien und Nachbearbeitung (LUT-Pfade, Symmetrie für rechte/linke Laufrichtung, Glätt-Radius).
- **Analyse:** Bestimmt Simulationsraum und Auswertungstiefe (Feldgröße, Ball-Position, Raster, Orientierungen, Max-Schritte).

Aktuelle Standardwerte im Studio:

- Epochen `20000`, Neuronen `128`, Batch-Größe `256`, CPU-Threads `0` (Auto)
- Glätt-Radius `3`, Symmetrie `rechts`
- Analyse: Feld `3.0m x 3.0m`, Ball `(1.5, 1.5)`, Raster `5cm`, Orientierungen `8`, Max-Schritte `200`

## Start

Im Repository-Root:

```bash
python BAI_Studio.py
```

## Abhängigkeiten

Die Studio-Datei nutzt u. a.:

- `PyQt5`
- `numpy`
- `torch`
- `matplotlib`

Zusätzlich nutzt `BAI_Studio.py` projektinterne Module bzw. Skripte:

- `Trainer_V2` – DQN-Modell, Zustandsberechnung und Training
- `generate_lut.py` – erzeugt LUT aus trainiertem Modell
- `LUT_Simplifier` – Symmetrie/Glättung und Header-Ausgabe
- `Analyser` – bewertet Modell-Erfolgsrate im Feldraster
- `LUT_Analyser` – bewertet LUT-Erfolgsrate im Feldraster

## Tests

Vorhandene Unit-Tests können so ausgeführt werden:

```bash
python -m unittest -v
```