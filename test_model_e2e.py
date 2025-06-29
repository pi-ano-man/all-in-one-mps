#!/usr/bin/env python3
"""
End-to-End Test für das MPS-kompatible allin1 Modell.

Dieses Skript testet den vollständigen Workflow von der Audioeingabe bis zur Modellausgabe.
"""

import torch
import json
import numpy as np
import soundfile as sf
import madmom
from pathlib import Path
from allin1_mps_model.manual_model import MPSCompatibleAllin1

def run_e2e_model_test():
    print("=== Starting Isolated End-to-End Model Test ===")
    print("This test verifies the complete pipeline from audio input to model output.")

    # === 1. Configuration and Paths ===
    BASE_DIR = Path(__file__).parent
    CONFIG_PATH = BASE_DIR / "hf_model_cache" / "config.json"
    WEIGHTS_PATH = BASE_DIR / "hf_model_cache" / "pytorch_model.bin"
    AUDIO_PATH = BASE_DIR / "audio" / "test_clip.wav"
    
    # Check if audio file exists
    if not AUDIO_PATH.exists():
        print(f"ERROR: Please place a test audio file at: {AUDIO_PATH}")
        print("You can use any WAV file, but a 15-30 second music clip is recommended.")
        return

    # === 2. Load and Prepare Model ===
    print("\n--> Loading Model...")
    try:
        with open(CONFIG_PATH, 'r') as f:
            model_config = json.load(f)
        
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = MPSCompatibleAllin1(model_config).to(device)
        
        # Try to load pre-trained weights
        try:
            print(f"Loading pre-trained weights from {WEIGHTS_PATH}...")
            # Lade die gesamte Datei
            loaded_data = torch.load(WEIGHTS_PATH, map_location=device)
            
            # Greife auf das state_dict innerhalb der geladenen Datei zu
            if 'state_dict' in loaded_data:
                state_dict = loaded_data['state_dict']
            else:
                # Wenn kein 'state_dict' Schlüssel existiert, verwende die gesamte Datei
                state_dict = loaded_data
                
            # Lade die Gewichte mit strict=True, um sicherzustellen, dass alle Schlüssel übereinstimmen
            model.load_state_dict(state_dict, strict=True)
            print("SUCCESS: Pre-trained weights loaded successfully!")
            model.eval()
        except Exception as e:
            print(f"WARNING: Could not load weights. Using random weights. Error: {e}")
    
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize model. Test aborted. Details: {e}")
        return

    # === 3. Load and Pre-process Audio ===
    print(f"\n--> Loading and Pre-processing Audio: {AUDIO_PATH}")
    try:
        # Load audio file
        waveform, sample_rate = sf.read(str(AUDIO_PATH), dtype='float32')
        
        # Convert to mono if needed
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        
        print(f"Audio loaded. Sample rate: {sample_rate}, Duration: {len(waveform)/sample_rate:.2f}s")

        # === STEP 3.2: THE ULTIMATE, EXPLICIT APPROACH ===
        
        # 3.2.1: Compute STFT (remains the same)
        stft_processor = madmom.audio.stft.STFT(
            waveform,
            sample_rate=sample_rate,
            fft_size=2048,
            hop_size=512
        )
        stft = np.array(stft_processor)  # -> (time, fft_bins)
        print(f"STFT shape: {stft.shape}, dtype: {stft.dtype}")
        
        # 3.2.2: Compute FFT bin frequencies to match STFT output
        # The STFT has 1024 frequency bins, so we need to create matching bin frequencies
        bin_frequencies = np.linspace(0, sample_rate/2, stft.shape[1])  # Match STFT frequency bins
        print(f"Bin frequencies shape: {bin_frequencies.shape}")
        
        # 3.2.3: MANUALLY DEFINE THE CENTER FREQUENCIES OF OUR 81 BANDS
        fmin = 30
        fmax = 17000
        band_frequencies = np.geomspace(fmin, fmax, num=81)  # NumPy's "logspace"
        print(f"Band frequencies shape: {band_frequencies.shape}")
        print(f"First 5 band frequencies: {band_frequencies[:5]}")
        
        # 3.2.4: USE THE CORRECT, FUNDAMENTAL FILTERBANK CLASS
        # LogFilterbank is the right choice from the available classes!
        filterbank = madmom.audio.filters.LogFilterbank(
            bin_frequencies=bin_frequencies,
            num_bands=81,  # Force exactly 81 frequency bands
            fmin=30,
            fmax=17000,
            norm_filters=True,
            unique_filters=True
        )
        print(f"Filterbank shape: {filterbank.shape} (should be [num_fft_bins, 81])")
        
        # 3.2.5: Apply the filterbank to the STFT
        spectrogram = np.dot(np.abs(stft)**2, filterbank)
        print(f"Spectrogram shape after filterbank: {spectrogram.shape} (Got {spectrogram.shape[1]} bands, need 81)")
        
        # 3.2.6: Force reduction to exactly 81 bands if needed
        if spectrogram.shape[1] != 81:
            print(f"Reducing from {spectrogram.shape[1]} to 81 bands using vectorized interpolation...")
            from scipy import interpolate
            
            # Use vectorized interpolation for efficiency
            original_indices = np.arange(spectrogram.shape[1])
            target_indices = np.linspace(0, spectrogram.shape[1]-1, 81)
            
            # Vectorized interpolation across all time frames at once
            f = interpolate.interp1d(original_indices, spectrogram, axis=1, kind='linear')
            spectrogram = f(target_indices)
            
            print(f"Final spectrogram shape: {spectrogram.shape} (MUST be [time, 81])")

        # 3.2.7: Transpose for the model (time, 81) -> (81, time)
        spectrogram = np.transpose(spectrogram, (1, 0))
        print(f"After transpose: {spectrogram.shape} (MUST be [81, time])")
        
        # 3.2.8: Create tensor and shape it for the model
        input_tensor = torch.from_numpy(spectrogram.copy()).float()
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(device)  # Add batch and stem dims
        input_tensor = input_tensor.repeat(1, 4, 1, 1)  # -> (1, 4, 81, time)

        # 3.2.9: Handle long sequences with chunking based on original implementation
        # Define chunking parameters based on the original config.yaml analysis
        HOP_SIZE_SAMPLES = 512  # Our madmom hop size
        SAMPLE_RATE = 44100     # Our madmom sample rate
        SEGMENT_SECONDS = 20    # Model trained on 20-second segments
        CHUNK_SIZE = int((SEGMENT_SECONDS * SAMPLE_RATE) / HOP_SIZE_SAMPLES)  # ~1723 steps
        OVERLAP_SECONDS = 2     # 2-second overlap for smooth transitions
        OVERLAP_STEPS = int((OVERLAP_SECONDS * SAMPLE_RATE) / HOP_SIZE_SAMPLES)
        HOP_SIZE_STEPS = CHUNK_SIZE - OVERLAP_STEPS
        
        print(f"Chunking parameters: CHUNK_SIZE={CHUNK_SIZE}, OVERLAP_STEPS={OVERLAP_STEPS}, HOP_SIZE_STEPS={HOP_SIZE_STEPS}")
        
        time_steps = input_tensor.shape[3]
        
        if time_steps > CHUNK_SIZE:
            print(f"Audio is too long ({time_steps} steps). Chunking into {CHUNK_SIZE}-step segments...")
            # For this test, we'll just take the first chunk
            input_tensor = input_tensor[:, :, :, :CHUNK_SIZE]
            print(f"Using first chunk: {input_tensor.shape}")
        
        print(f"SUCCESS: Pre-processing complete. Final input tensor shape: {input_tensor.shape}")
        print(f"Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}], "
              f"Mean: {input_tensor.mean():.3f}, Std: {input_tensor.std():.3f}")
        print(f"First few values: {input_tensor.flatten()[:5].tolist()}")
        
    except Exception as e:
        print(f"FATAL ERROR: Could not pre-process audio. Test aborted. Details: {e}")
        import traceback
        traceback.print_exc()
        return

    # === 4. Run Model Inference ===
    print("\n--> Running Model Inference...")
    try:
        with torch.no_grad():
            outputs = model(input_tensor)
        
        print("\n=== TEST COMPLETE ===")
        print("SUCCESS: Forward pass completed successfully!")
        print("\nModel Outputs:")
        def print_outputs(outputs, indent=0):
            for key, value in outputs.items():
                if isinstance(value, dict):
                    print(f"{'  ' * indent}{key}:")
                    print_outputs(value, indent + 1)
                else:
                    print(f"{'  ' * indent}{key}: {value.shape} | {value.dtype}")
        
        print_outputs(outputs)
        
        # Print some sample values
        print("\nSample Output Values:")
        def print_sample_values(outputs, indent=0):
            for key, value in outputs.items():
                if isinstance(value, dict):
                    print(f"{'  ' * indent}{key}:")
                    print_sample_values(value, indent + 1)
                else:
                    if value.numel() > 5:
                        sample = value.flatten()[:5].tolist()
                        print(f"{'  ' * indent}{key}: {sample}... (shape: {value.shape})")
                    else:
                        print(f"{'  ' * indent}{key}: {value.tolist()}")
        
        print_sample_values(outputs)
    
    except Exception as e:
        print(f"ERROR: Model inference failed. Details: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    run_e2e_model_test()
