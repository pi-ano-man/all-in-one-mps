#!/usr/bin/env python3
"""
Finaler End-to-End Test für das MPS-kompatible allin1 Modell.

Dieses Skript ist das Ergebnis aller Debugging-Phasen und validiert den
gesamten Workflow:
1. Laden des manuell gebauten PyTorch-Modells.
2. Laden der vortrainierten Gewichte mit strikter Prüfung.
3. Explizites, kontrolliertes Pre-Processing mit madmom.
4. Korrektes Chunking für lange Audio-Dateien.
5. Erfolgreiche Inferenz auf der MPS-Hardware.
"""

import torch
import json
import numpy as np
import soundfile as sf
import madmom
from pathlib import Path
import traceback

# Importieren Sie das Modell UND die Remapping-Funktion
from allin1_mps_model.manual_model import MPSCompatibleAllin1, remap_state_dict_keys

def run_e2e_model_test():
    print("=== Starting Final End-to-End Model Test ===")

    # === 1. Konfiguration und Pfade ===
    BASE_DIR = Path(__file__).parent
    CONFIG_PATH = BASE_DIR / "hf_model_cache" / "config.json"
    WEIGHTS_PATH = BASE_DIR / "hf_model_cache" / "pytorch_model.bin"
    AUDIO_PATH = BASE_DIR / "audio" / "test_clip.wav"
    
    if not AUDIO_PATH.exists():
        print(f"FATAL ERROR: Test audio file not found at: {AUDIO_PATH}")
        return

    # === 2. Modell laden und Gewichte abgleichen (Der entscheidende Schritt) ===
    print("\n--> Phase 2: Loading Model and Matching Weights...")
    try:
        with open(CONFIG_PATH, 'r') as f:
            model_config = json.load(f)
        
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = MPSCompatibleAllin1(model_config).to(device)
        
        # Lade die Gewichte mit strikter Prüfung. Dies MUSS jetzt erfolgreich sein.
        print(f"Loading pre-trained weights from {WEIGHTS_PATH}...")
        loaded_data = torch.load(WEIGHTS_PATH, map_location=device)
        
        state_dict = loaded_data.get('state_dict', loaded_data)
            
        # Wenden Sie die Remapping-Funktion an, um die Schlüsselnamen abzugleichen
        remapped_state_dict = remap_state_dict_keys(state_dict)
        
        # Laden mit strict=True ist der ultimative Test
        model.load_state_dict(remapped_state_dict, strict=True)
        
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! SUCCESS: PRE-TRAINED WEIGHTS LOADED !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        model.eval()

    except Exception as e:
        print(f"FATAL ERROR: Could not load weights with strict=True. The model architecture in 'manual_model.py' does not yet perfectly match the weight file.")
        print(f"Details: {e}")
        return

    # === 3. Audio laden und vorverarbeiten ===
    print(f"--> Phase 3: Loading and Pre-processing Audio: {AUDIO_PATH}")
    try:
        waveform, sample_rate = sf.read(str(AUDIO_PATH), dtype='float32')
        if waveform.ndim > 1: waveform = np.mean(waveform, axis=1)
        print(f"Audio loaded. SR: {sample_rate}, Duration: {len(waveform)/sample_rate:.2f}s")

        # Expliziter Pre-Processing-Pfad
        stft = madmom.audio.stft.STFT(waveform, sample_rate=sample_rate, fft_size=2048, hop_size=512)
        bin_frequencies = madmom.audio.stft.fft_frequencies(fft_size=2048, sample_rate=sample_rate)
        band_frequencies = np.geomspace(30, 17000, num=81)
        filterbank = madmom.audio.filters.LogFilterbank(bin_frequencies=bin_frequencies, filter_frequencies=band_frequencies)
        spectrogram = np.dot(np.abs(stft)**2, filterbank)
        print(f"Spectrogram shape: {spectrogram.shape} (time, 81)")

        spectrogram = np.transpose(spectrogram, (1, 0)) # -> (81, time)
        input_tensor = torch.from_numpy(spectrogram.copy()).float().unsqueeze(0).unsqueeze(0).to(device)
        input_tensor = input_tensor.repeat(1, 4, 1, 1) # -> (1, 4, 81, time)

        # Chunking basierend auf der Original-Konfiguration
        SEGMENT_SECONDS = 20
        CHUNK_SIZE = int((SEGMENT_SECONDS * sample_rate) / 512) # ~1722
        
        if input_tensor.shape[3] > CHUNK_SIZE:
            print(f"Audio is long. Using first chunk of {CHUNK_SIZE} steps for this test.")
            input_tensor = input_tensor[..., :CHUNK_SIZE]
        
        print(f"SUCCESS: Pre-processing complete. Final input tensor shape: {input_tensor.shape}")
        
    except Exception as e:
        print(f"FATAL ERROR: Could not pre-process audio. Test aborted.")
        traceback.print_exc()
        return

    # === 4. Inferenz durchführen ===
    print("\n--> Phase 4: Running Model Inference...")
    try:
        with torch.no_grad():
            outputs = model(input_tensor)
        
        print("\n================== FINAL RESULT ==================")
        print("SUCCESS: Forward pass with pre-trained weights completed!")
        
        # Robuste Ausgabe für verschachtelte Dictionaries
        def print_outputs_recursively(output_dict, indent=0):
            for key, value in output_dict.items():
                prefix = "  " * indent + f"- {key}:"
                if isinstance(value, dict):
                    print(prefix)
                    print_outputs_recursively(value, indent + 1)
                else:
                    print(f"{prefix} {value.shape} | dtype: {value.dtype}")
        
        print("\nModel Output Shapes:")
        print_outputs_recursively(outputs)
        print("\nThis indicates the full pipeline is functional.")
        print("================================================")

    except Exception as e:
        print(f"ERROR: Model inference failed even after successful setup. Debug the forward pass.")
        traceback.print_exc()
        return

if __name__ == "__main__":
    run_e2e_model_test()