#!/usr/bin/env python3
import torch
import sys
from pathlib import Path

from allin1_mps_model.manual_model import MPSCompatibleAllin1, remap_state_dict_keys


def analyze_architecture():
    weights_path = Path("./hf_model_cache/pytorch_model.bin")
    print(f"Using weights from: {weights_path}")
    
    pretrained_sd = torch.load(weights_path, map_location='cpu').get('state_dict')
    
    print("\nCreating model instance...")
    config = {'num_attention_heads': 4, 'num_hidden_layers': 11}
    model = MPSCompatibleAllin1(config)
    
    print("\nApplying key remapping...")
    remapped_sd = remap_state_dict_keys(pretrained_sd)

    print("\nATTEMPTING TO LOAD FINAL STATE DICT...")
    try:
        model.load_state_dict(remapped_sd, strict=True)
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!               VICTORY! VICTORY! VICTORY!               !!!")
        print("!!!  THE WEIGHTS HAVE BEEN LOADED SUCCESSFULLY (strict=True) !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        print("Das Projekt ist ERFOLGREICH. Das Modell ist bereit für die Inferenz.")
    except RuntimeError as e:
        print("\n------------------FINAL CHECKLIST------------------")
        print("FEHLSCHLAG. Fast geschafft. Dies sind die letzten verbleibenden Fehler:")
        print(e)
        print("-------------------------------------------------")


if __name__ == "__main__":
    analyze_architecture()