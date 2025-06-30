# Archiv - allin1 MPS Debugging-Phase

## Zweck dieses Archivs
Dieses Verzeichnis enthält alle Dateien aus der intensiven Debugging- und Reverse-Engineering-Phase des allin1 MPS-Projekts (Juni 2025).

## Inhalt

### debug_analysis/
Enthält alle Analyse- und Debug-Skripte, die während der Modell-Portierung verwendet wurden:

- **inspect_weights.py** - Skript zur Analyse der pretrained weights Struktur
- **validate_architecture.py** - Validierung der Modellarchitektur und Key-Mapping
- **weight_keys_flat.txt** - Flache Liste aller Keys im pretrained model
- **weight_structure_tree.txt** - Hierarchische Darstellung der Modellstruktur

## Warum archiviert?
Diese Dateien waren essentiell für das Reverse Engineering, werden aber für den produktiven Betrieb nicht mehr benötigt. Sie sind hier archiviert für:
- Zukünftige Referenz bei ähnlichen Problemen
- Dokumentation des Debugging-Prozesses
- Falls wir nochmal tiefer in die Modellstruktur schauen müssen

## Hinweis
Das funktionierende Ergebnis dieser Arbeit befindet sich in:
- `allin1_mps_model/manual_model.py` - Das finale MPS-kompatible Modell
