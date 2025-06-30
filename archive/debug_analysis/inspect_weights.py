#!/usr/bin/env python3
"""
Diagnose-Skript zur Analyse der Gewichtsdatei-Struktur.

Dieses Skript lädt die vortrainierten Gewichte und gibt eine detaillierte
Übersicht über die Modellarchitektur basierend auf den state_dict-Schlüsseln.
"""

import torch
import os
from collections import defaultdict
from pathlib import Path

def build_tree(keys):
    """
    Erstellt eine hierarchische Baumstruktur aus den state_dict-Schlüsseln.
    
    Args:
        keys: Liste von state_dict-Schlüsseln
        
    Returns:
        dict: Verschachtelte Wörterbuchstruktur
    """
    tree = {}
    for key in keys:
        parts = key.split('.')
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = None  # Mark leaf node
    return tree

def print_tree(node, prefix=''):
    """
    Gibt die Baumstruktur formatiert aus.
    
    Args:
        node: Aktueller Knoten im Baum
        prefix: Präfix für die Einrückung
    """
    if not isinstance(node, dict):
        return
        
    for i, (key, child) in enumerate(sorted(node.items())):
        # Bestimme das Präfix für die nächste Ebene
        if i == len(node) - 1:
            new_prefix = prefix + '    '
            print(f"{prefix}└── {key}")
        else:
            new_prefix = prefix + '│   '
            print(f"{prefix}├── {key}")
        
        # Rekursiver Aufruf für Kinderknoten
        print_tree(child, new_prefix)

def main():
    # Pfad zur Gewichtsdatei
    weights_path = Path("hf_model_cache/pytorch_model.bin")
    
    print(f"Lade Gewichte aus: {weights_path.absolute()}")
    
    try:
        # Lade die Gewichte
        if not weights_path.exists():
            raise FileNotFoundError(f"Gewichtsdatei nicht gefunden: {weights_path}")
            
        # Lade die Gewichte auf die CPU
        loaded_data = torch.load(weights_path, map_location=torch.device('cpu'))
        
        # Extrahiere das state_dict
        if 'state_dict' in loaded_data:
            state_dict = loaded_data['state_dict']
            print(f"Geladenes state_dict mit {len(state_dict)} Einträgen")
        else:
            state_dict = loaded_data
            print(f"Kein 'state_dict' gefunden, verwende direkt geladene Daten mit {len(state_dict)} Einträgen")
        
        # Extrahiere alle Schlüssel
        keys = list(state_dict.keys())
        print(f"\nErste 10 Schlüssel:")
        for key in keys[:10]:
            print(f"  - {key}: {state_dict[key].shape}")
        
        # Baue die Baumstruktur
        print("\nModellarchitektur (Baumansicht):")
        tree = build_tree(keys)
        print_tree(tree)
        
        # Zähle die Vorkommen jedes Präfixes
        print("\nHäufigste Präfixe:")
        prefix_counts = defaultdict(int)
        for key in keys:
            parts = key.split('.')
            for i in range(1, len(parts)):
                prefix = '.'.join(parts[:i])
                prefix_counts[prefix] += 1
        
        for prefix, count in sorted(prefix_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"  - {prefix}: {count} Parameter")
        
        # Suche nach spezifischen Mustern
        print("\nWichtige Layer-Typen:")
        layer_types = {}
        for key in keys:
            if 'weight' in key or 'bias' in key:
                layer_type = key.split('.')[-2] if '.' in key else 'root'
                layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        for layer_type, count in sorted(layer_types.items(), key=lambda x: -x[1]):
            print(f"  - {layer_type}: {count} Vorkommen")
        
    except Exception as e:
        print(f"Fehler beim Analysieren der Gewichte: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
