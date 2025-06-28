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
                # Entferne 'module.' Präfix, falls vorhanden (für DataParallel/DPT)
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                print("SUCCESS: Pre-trained weights loaded successfully!")
                model.eval()
            else:
                # Falls die Datei direkt das state_dict enthält
                model.load_state_dict(loaded_data)
                print("SUCCESS: Pre-trained weights loaded successfully! (direct state_dict)")
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
        
        # 3.2.2: Compute FFT bin frequencies
        # The STFT has 1024 bins, so we need to match that
        actual_fft_bins = stft.shape[1]  # 1024
        bin_frequencies = madmom.audio.stft.fft_frequencies(actual_fft_bins * 2, sample_rate)
        # Take only the first 1024 bins (positive frequencies)
        bin_frequencies = bin_frequencies[:actual_fft_bins]
        print(f"Bin frequencies shape: {bin_frequencies.shape} (should be {actual_fft_bins})")
        
        # 3.2.3: Create a custom filterbank matrix with exactly 81 bands
        print("Creating custom filterbank with exactly 81 logarithmic bands...")
        
        # Define 81 logarithmically spaced center frequencies
        fmin, fmax = 30.0, 17000.0
        center_freqs = np.geomspace(fmin, fmax, num=81)
        
        # Create the filterbank matrix (num_fft_bins x num_bands)
        num_fft_bins = len(bin_frequencies)
        filterbank_matrix = np.zeros((num_fft_bins, 81))
        
        # Create triangular filters for each band
        for i, center_freq in enumerate(center_freqs):
            # Calculate filter edges (triangular filter)
            if i == 0:
                f_low = center_freq
            else:
                f_low = center_freqs[i-1]
                
            if i == len(center_freqs) - 1:
                f_high = center_freq
            else:
                f_high = center_freqs[i+1]
            
            # Create triangular response
            for j, bin_freq in enumerate(bin_frequencies):
                if f_low <= bin_freq <= f_high:
                    if bin_freq <= center_freq:
                        # Rising edge
                        if center_freq != f_low:
                            filterbank_matrix[j, i] = (bin_freq - f_low) / (center_freq - f_low)
                        else:
                            filterbank_matrix[j, i] = 1.0
                    else:
                        # Falling edge
                        if f_high != center_freq:
                            filterbank_matrix[j, i] = (f_high - bin_freq) / (f_high - center_freq)
                        else:
                            filterbank_matrix[j, i] = 1.0
        
        # Normalize each filter
        for i in range(81):
            if np.sum(filterbank_matrix[:, i]) > 0:
                filterbank_matrix[:, i] /= np.sum(filterbank_matrix[:, i])
        
        # Create the madmom Filterbank object
        filterbank = madmom.audio.filters.Filterbank(
            data=filterbank_matrix,
            bin_frequencies=bin_frequencies
        )
        
        print(f"Filterbank shape: {filterbank_matrix.shape} (should be [{num_fft_bins}, 81])")
        print(f"Center frequencies: {center_freqs[:5]}...{center_freqs[-5:]}")
        
        # 3.2.5: Apply the filterbank to the STFT
        # Use the filterbank matrix directly for the dot product
        spectrogram = np.dot(np.abs(stft)**2, filterbank_matrix)
        print(f"Spectrogram shape after filterbank: {spectrogram.shape} (MUST be [time, 81])")

        # 3.2.6: Transpose for the model (time, 81) -> (81, time)
        spectrogram = np.transpose(spectrogram, (1, 0))
        print(f"After transpose: {spectrogram.shape} (MUST be [81, time])")
        
        # 3.2.7: Create tensor and shape it for the model
        input_tensor = torch.from_numpy(spectrogram.copy()).float()
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(device)  # Add batch and stem dims
        input_tensor = input_tensor.repeat(1, 4, 1, 1)  # -> (1, 4, 81, time)

        print(f"SUCCESS: Pre-processing complete. Input shape: {input_tensor.shape}")
        print(f"Input range: [{torch.min(input_tensor):.3f}, {torch.max(input_tensor):.3f}], "
              f"Mean: {torch.mean(input_tensor):.3f}, Std: {torch.std(input_tensor):.3f}")
        print(f"First few values: {input_tensor[0, 0, 0, :5].tolist()}\n")

        # --- Chunking Parameters ---
        CHUNK_SIZE = 4096  # Must be less than model's max sequence length (5000)
        HOP_SIZE = 3584    # Overlap = CHUNK_SIZE - HOP_SIZE = 512 samples
        MIN_CHUNK_SIZE = 128  # Skip chunks smaller than this
        
        # Get total number of time steps
        total_steps = input_tensor.shape[-1]
        num_chunks = (total_steps + HOP_SIZE - 1) // HOP_SIZE  # Ceiling division
        
        print(f"Processing {total_steps} time steps in {num_chunks} chunks...")
        print(f"Chunk size: {CHUNK_SIZE}, Hop size: {HOP_SIZE}, Overlap: {CHUNK_SIZE - HOP_SIZE}")
        
        # Store outputs for each chunk
        all_chunk_outputs = []
        
        # Process each chunk
        with torch.no_grad():
            for i in range(0, total_steps, HOP_SIZE):
                # Calculate chunk boundaries
                chunk_start = i
                chunk_end = min(i + CHUNK_SIZE, total_steps)
                chunk = input_tensor[..., chunk_start:chunk_end]
                
                # Skip if chunk is too small
                if chunk.shape[-1] < MIN_CHUNK_SIZE:
                    print(f"Skipping small chunk: {chunk.shape[-1]} samples")
                    continue
                    
                print(f"Processing chunk: steps {chunk_start} to {chunk_end-1} (size: {chunk.shape[-1]})")
                
                # Process chunk
                chunk_output = model(chunk)
                all_chunk_outputs.append((chunk_start, chunk_output))
        
        # --- Stitching Logic ---
        if not all_chunk_outputs:
            raise ValueError("No valid chunks were processed")
            
        print("\nStitching chunks together...")
        
        # Initialize stitched outputs dictionary
        stitched_outputs = {}
        output_heads = all_chunk_outputs[0][1].keys()  # Get output head names
        
        for head in output_heads:
            # Get feature dimension from first chunk
            feature_dim = all_chunk_outputs[0][1][head].shape[-1]
            
            # Create output tensor with proper shape
            # Shape: (batch * stems, total_steps, feature_dim)
            batch_stems = all_chunk_outputs[0][1][head].shape[0]
            full_output = torch.zeros((batch_stems, total_steps, feature_dim), 
                                   device=input_tensor.device)
            
            # Create overlap-add window (simple linear fade)
            window = torch.ones(CHUNK_SIZE, device=input_tensor.device)
            if CHUNK_SIZE > HOP_SIZE:
                # Create fade in/out for overlapping regions
                overlap = CHUNK_SIZE - HOP_SIZE
                window[:overlap] = torch.linspace(0, 1, overlap, device=input_tensor.device)
                window[-overlap:] = torch.linspace(1, 0, overlap, device=input_tensor.device)
            
            # Stitch chunks together with overlap-add
            for chunk_start, chunk_output in all_chunk_outputs:
                chunk_data = chunk_output[head]
                chunk_size = chunk_data.shape[1]
                chunk_end = chunk_start + chunk_size
                
                # Apply window
                chunk_data = chunk_data * window[:chunk_size]
                
                # Add to output with overlap-add
                full_output[..., chunk_start:chunk_end, :] += chunk_data
            
            stitched_outputs[head] = full_output
        
        print("\nStitching complete!")
        print("\nStitched output shapes:")
        for key, value in stitched_outputs.items():
            print(f"  {key}: {value.shape}")
            
        # For backward compatibility, assign to outputs
        outputs = stitched_outputs

    except Exception as e:
        print(f"FATAL ERROR: Could not run model inference. Test aborted. Details: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        if 'all_chunk_outputs' in locals():
            print(f"Number of processed chunks: {len(all_chunk_outputs)}")
        raise

    # === 4. Run Model Inference ===
    print("\n--> Running Model Inference...")
    try:
        print("\n=== TEST COMPLETE ===")
        print("SUCCESS: Forward pass completed successfully!")
        print("\nModel Outputs:")
        for key, value in outputs.items():
            print(f"  {key}: {value.shape} | {value.dtype}")
        
        # Print some sample values
        print("\nSample Output Values:")
        for key, value in outputs.items():
            if key == 'functional':
                # For functional, show class with max probability at first time step
                max_vals, max_idxs = torch.max(value[0, 0], dim=0)
                print(f"  {key}: Class {max_idxs.item()} with probability {max_vals.item():.4f}")
            else:
                # For other outputs, just show first few values
                print(f"  {key}: {value[0, :5].cpu().numpy()}")
    
    except Exception as e:
        print(f"ERROR: Model inference failed. Details: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    run_e2e_model_test()
