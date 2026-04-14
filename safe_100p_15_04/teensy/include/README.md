# robot_lut.h

Dieses Verzeichnis enthält die vorberechnete Lookup-Tabelle `robot_lut.h`,
die automatisch von `generate_lut.py` aus dem trainierten Modell
`roboter_rl_modell_128.pth` erzeugt wurde.

## Datei neu erzeugen

Wenn du ein anderes Modell oder andere Neuronen verwendest, erzeuge die Datei
neu und kopiere sie hierher:

```bash
# Im Projekt-Stammverzeichnis (wo generate_lut.py liegt):
python generate_lut.py --neuronen 128

# Die fertige Datei dann hierher kopieren:
cp robot_lut.h teensy/include/robot_lut.h
```

Danach `pio run --target upload` ausführen (siehe TEENSY_ANLEITUNG.md).
