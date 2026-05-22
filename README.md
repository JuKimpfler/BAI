# BAI Studio

Diese Repository-Version ist auf das zentrale Desktop-Tool **`BAI_Studio.py`** ausgerichtet.  
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

- **Training:** Epochen, Neuronen (64/128/256/400), Batch-Größe, CPU-Threads, Modelldatei
- **Epsilon-Schedule:** frei definierbarer Verlauf im Format `Prozent:Wert`
- **LUT:** Pfad zur LUT, Pfad zur optimierten LUT, Symmetrie-Seite (`rechts`/`links`), Glätt-Radius
- **Analyse:** Feldgröße, Ball-Position, Grid-Schrittweite, Orientierungen, maximale Schritte

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

Zusätzlich werden Projektmodule wie Trainer/Analyser/LUT-Komponenten direkt importiert und im Workflow aufgerufen.

## Tests

Vorhandene Unit-Tests können so ausgeführt werden:

```bash
python -m unittest -v
```