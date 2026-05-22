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
- **Epsilon-Schedule:** Definiert den Explorationsverlauf im DQN-Training als Zeilenliste `Trainingsfortschritt in %:Epsilon-Wert` (z. B. `0:1.0`, `50:0.5`, `100:0.02`).
- **LUT:** Steuert Ein-/Ausgabedateien und Nachbearbeitung (LUT-Pfade, Symmetrie für rechte/linke Laufrichtung, Glätt-Radius).
- **Analyse:** Bestimmt Simulationsraum und Auswertungstiefe (Feldgröße, Ball-Position, Raster, Orientierungen, Max-Schritte).

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

- `Analyser`
- `LUT_Analyser`
- `LUT_Simplifier`
- `Trainer_V2`
- `generate_lut.py`

## Tests

Vorhandene Unit-Tests können so ausgeführt werden:

```bash
python -m unittest -v
```