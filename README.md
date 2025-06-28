# allin1 MPS Testbench

Isolierte Testumgebung für das MPS-kompatible allin1 Modell.

## Einrichtung

1. Virtuelle Umgebung erstellen und aktivieren:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Auf Windows: venv\Scripts\activate
   ```

2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

3. Test-Audiodatei hinzufügen:
   - Legen Sie eine kurze Audiodatei (15-30 Sekunden) als `test_clip.wav` im `audio`-Verzeichnis ab.

## Ausführung

Führen Sie den End-to-End-Test mit folgendem Befehl aus:

```bash
python test_model_e2e.py
```

## Ordnerstruktur

```
allin1-mps-testbench/
├── venv/                      # Virtuelle Python-Umgebung
├── hf_model_cache/            # Modellkonfiguration und -gewichte
│   └── config.json
├── audio/                     # Test-Audiodateien
│   └── test_clip.wav
├── allin1_mps_model/          # Python-Paket mit der MPS-kompatiblen Implementierung
│   ├── __init__.py
│   └── manual_model.py
├── requirements.txt           # Python-Abhängigkeiten
└── test_model_e2e.py          # Haupttestskript
```

## Nächste Schritte

1. Sicherstellen, dass die Test-Audiodatei vorhanden ist
2. Das Skript `test_model_e2e.py` ausführen
3. Die Ausgabe auf Fehler überprüfen
4. Bei Erfolg: Integration in die Hauptanwendung
