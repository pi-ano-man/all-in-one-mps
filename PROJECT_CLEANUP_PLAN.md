# Aufräumplan für allin1-MPS Projekt

## 🚀 Produktionsdateien (BEHALTEN)

### Kernmodell
- `allin1_mps_model/manual_model.py` - **UNSER HERZSTÜCK**: Das MPS-kompatible Modell
- `allin1_mps_model/__init__.py` - Package-Initialisierung

### Wichtige Abhängigkeiten
- `hf_model_cache/pytorch_model.bin` - Die trainierten Gewichte
- `hf_model_cache/[andere Dateien]` - Modell-Cache
- `requirements.txt` - Python-Abhängigkeiten
- `README.md` - Projektdokumentation

### Audio-Testdateien
- `audio/` - Testmusik für End-to-End Tests

## 🧹 Debug/Analyse-Dateien (ZUM LÖSCHEN)

### Temporäre Analyse-Skripte
- `inspect_weights.py` - War nur zur Gewichtsanalyse
- `validate_architecture.py` - War nur zur Architektur-Validierung
- `weight_keys_flat.txt` - Debug-Output der Gewichtsnamen
- `weight_structure_tree.txt` - Debug-Output der Modellstruktur

### Python-Cache
- `allin1_mps_model/__pycache__/` - Python-Bytecode (wird automatisch neu generiert)

### Virtuelle Umgebung
- `venv/` - Kann jederzeit mit `pip install -r requirements.txt` neu erstellt werden

## ❓ Zu überprüfen

### test_model_e2e.py
- **Status**: Möglicherweise nützlich für zukünftige Tests
- **Empfehlung**: Behalten, aber eventuell anpassen für Produktionsnutzung

## 📋 Empfohlene Schritte

1. **Backup erstellen** (zur Sicherheit):
   ```bash
   tar -czf allin1_mps_backup_$(date +%Y%m%d).tar.gz .
   ```

2. **Debug-Dateien löschen**:
   ```bash
   rm inspect_weights.py validate_architecture.py weight_*.txt
   rm -rf allin1_mps_model/__pycache__
   rm -rf venv
   ```

3. **Git aufräumen** (optional):
   ```bash
   git rm --cached venv  # Falls versehentlich committed
   echo "venv/" >> .gitignore
   echo "__pycache__/" >> .gitignore
   echo "*.pyc" >> .gitignore
   ```

## 🎯 Nächste Schritte nach dem Aufräumen

1. **Integration in die Hauptanwendung**: 
   - Das `manual_model.py` in die eigentliche allin1-Pipeline einbauen
   - Den NATTEN-Transformer durch unser MPS-Modell ersetzen

2. **Performance-Tests**:
   - Geschwindigkeitsvergleich NATTEN vs. MPS-Modell
   - Memory-Profiling auf MPS

3. **Dokumentation**:
   - README.md aktualisieren mit Installationsanweisungen
   - Nutzungsbeispiele hinzufügen
