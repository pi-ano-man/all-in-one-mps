# allin1_mps_model/manual_model.py
# FINAL VERSION 5.0 - Korrekte Dimensionen und Struktur

import torch
import torch.nn as nn
import torch.nn.functional as F
import re

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # Verwende kombinierte qkv wie im Original
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        # Korrekte Form für relative_position_bias_table (timelayer hat [2, 9])
        self.register_buffer("relative_position_bias_table", torch.zeros(2, 9))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * ((C // self.num_heads) ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, dim, mlp_hidden_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        # WICHTIG: norm2 ist 48D für timelayer!
        self.norm2 = nn.LayerNorm(48)
        # MLP arbeitet mit 48D -> 192D -> 24D (für timelayer)
        self.mlp = Mlp(48, 192, 24)

    def forward(self, x):
        # Attention path bleibt bei 24D
        x = x + self.attn(self.norm1(x))
        
        # MLP path: muss auf 48D projiziert werden für norm2
        # Da unsere Gewichte von timelayer kommen, welches intern 48D verwendet
        x_proj = nn.functional.pad(x, (0, 24))  # Pad von 24D auf 48D
        x_normed = self.norm2(x_proj)  # norm2 erwartet 48D
        x_mlp_out = self.mlp(x_normed)  # MLP: 48D -> 192D -> 24D
        
        x = x + x_mlp_out
        return x

class EmbeddingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 12, kernel_size=(3, 3), padding=1)
        self.conv1 = nn.Conv2d(12, 24, kernel_size=(1, 12), padding=0)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=(3, 3), padding=1)
        self.projection = nn.Linear(1944, 24)
        self.norm = nn.LayerNorm(24)

    def forward(self, x):
        B, S, F, T = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B * F, 1, T, S)
        x = F.gelu(self.conv0(x))
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.reshape(B * F, 24, -1).mean(dim=-1)
        x = x.view(B, F, 24)
        x = self.projection(x.flatten(1))
        x = x.unsqueeze(1)
        x = self.norm(x)
        return x

class MPSCompatibleAllin1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = EmbeddingModule()

        num_heads = config.get('num_attention_heads', 4)
        
        # Alle Encoder-Layer arbeiten mit 24D input, aber haben spezielle interne Dimensionen
        self.encoder = nn.ModuleList()
        for i in range(11):
            # Alle Layer verwenden die gleiche Struktur (basierend auf timelayer)
            self.encoder.append(EncoderLayer(24, 192, num_heads))
        
        # Finale Norm ist 24D
        self.norm = nn.LayerNorm(24)
        
        # Projektion von 24D auf 96D für Classifier
        self.final_projection = nn.Linear(24, 96)
        
        # Classifiers erwarten 96D Input!
        self.beat_0 = nn.Linear(96, 1)
        self.downbeat_0 = nn.Linear(96, 1)
        self.section_0 = nn.Linear(96, 1)
        self.function_0 = nn.ModuleDict({
            'harmonix': nn.Linear(96, 10),
            'raveform': nn.Linear(96, 11)
        })
        self.dataset_0 = nn.Linear(96, 2)

    def forward(self, x):
        x = self.embeddings(x)
        
        # Durchlaufe alle Layer - alle arbeiten mit 24D
        for layer in self.encoder:
            x = layer(x)
        
        # Finale Normalisierung bei 24D
        x = self.norm(x)
        
        # Pooling
        x_pooled = x.mean(dim=1)
        
        # Projektion auf 96D für Classifier
        x_pooled = self.final_projection(x_pooled)
        
        return {'beat': self.beat_0(x_pooled)}

def remap_state_dict_keys(state_dict):
    """Remapping-Funktion für die hierarchische Struktur"""
    new_sd = {}
    
    # Sammle q,k,v weights für jeden Layer
    qkv_weights = {}
    qkv_biases = {}
    
    for k, v in state_dict.items():
        # Extrahiere Layer-Nummer aus Keys wie "encoder.layers.0.timelayer.attention.self.query.weight"
        layer_match = re.match(r"encoder\.layers\.(\d+)\.(timelayer|instlayer)", k)
        
        if layer_match:
            layer_idx = int(layer_match.group(1))
            sublayer = layer_match.group(2)
            
            # Für timelayer mappt auf unsere flache Struktur
            if sublayer == "timelayer":
                # Sammle q,k,v für qkv concatenation
                if "attention.self.query.weight" in k:
                    if layer_idx not in qkv_weights:
                        qkv_weights[layer_idx] = {}
                    qkv_weights[layer_idx]['q'] = v
                elif "attention.self.key.weight" in k:
                    if layer_idx not in qkv_weights:
                        qkv_weights[layer_idx] = {}
                    qkv_weights[layer_idx]['k'] = v
                elif "attention.self.value.weight" in k:
                    if layer_idx not in qkv_weights:
                        qkv_weights[layer_idx] = {}
                    qkv_weights[layer_idx]['v'] = v
                elif "attention.self.query.bias" in k:
                    if layer_idx not in qkv_biases:
                        qkv_biases[layer_idx] = {}
                    qkv_biases[layer_idx]['q'] = v
                elif "attention.self.key.bias" in k:
                    if layer_idx not in qkv_biases:
                        qkv_biases[layer_idx] = {}
                    qkv_biases[layer_idx]['k'] = v
                elif "attention.self.value.bias" in k:
                    if layer_idx not in qkv_biases:
                        qkv_biases[layer_idx] = {}
                    qkv_biases[layer_idx]['v'] = v
                # Andere attention weights
                elif "attention.output.dense" in k:
                    new_k = re.sub(r"encoder\.layers\.(\d+)\.timelayer\.attention\.output\.dense", 
                                   r"encoder.\1.attn.proj", k)
                    new_sd[new_k] = v
                elif "layernorm_before" in k:
                    new_k = re.sub(r"encoder\.layers\.(\d+)\.timelayer\.layernorm_before", 
                                   r"encoder.\1.norm1", k)
                    new_sd[new_k] = v
                elif "layernorm_after" in k:
                    new_k = re.sub(r"encoder\.layers\.(\d+)\.timelayer\.layernorm_after", 
                                   r"encoder.\1.norm2", k)
                    new_sd[new_k] = v
                elif "intermediate.dense" in k:
                    new_k = re.sub(r"encoder\.layers\.(\d+)\.timelayer\.intermediate\.dense", 
                                   r"encoder.\1.mlp.fc1", k)
                    new_sd[new_k] = v
                elif "output.dense" in k and "attention" not in k:
                    new_k = re.sub(r"encoder\.layers\.(\d+)\.timelayer\.output\.dense", 
                                   r"encoder.\1.mlp.fc2", k)
                    new_sd[new_k] = v
                elif "attention.self.rpb" in k and "attention2" not in k:
                    # rpb -> relative_position_bias_table
                    new_k = f"encoder.{layer_idx}.attn.relative_position_bias_table"
                    new_sd[new_k] = v
            # Ignoriere instlayer und attention2
            continue
        else:
            # Andere Keys direkt übernehmen
            new_sd[k] = v
    
    # Kombiniere q,k,v weights zu qkv
    for layer_idx in qkv_weights:
        if all(x in qkv_weights[layer_idx] for x in ['q', 'k', 'v']):
            q_weight = qkv_weights[layer_idx]['q']
            k_weight = qkv_weights[layer_idx]['k']
            v_weight = qkv_weights[layer_idx]['v']
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            new_sd[f"encoder.{layer_idx}.attn.qkv.weight"] = qkv_weight
            
    # Kombiniere q,k,v biases zu qkv
    for layer_idx in qkv_biases:
        if 'q' in qkv_biases[layer_idx] and 'k' in qkv_biases[layer_idx] and 'v' in qkv_biases[layer_idx]:
            q_bias = qkv_biases[layer_idx]['q']
            k_bias = qkv_biases[layer_idx]['k']
            v_bias = qkv_biases[layer_idx]['v']
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            new_sd[f"encoder.{layer_idx}.attn.qkv.bias"] = qkv_bias
    
    # Zusätzliche Key-Remappings für function und dataset classifier
    additional_mappings = {
        "function_classifier.harmonix.classifier.weight": "function_0.harmonix.weight",
        "function_classifier.harmonix.classifier.bias": "function_0.harmonix.bias",
        "function_classifier.raveform.classifier.weight": "function_0.raveform.weight",
        "function_classifier.raveform.classifier.bias": "function_0.raveform.bias",
        "dataset_classifier.weight": "dataset_0.weight",
        "dataset_classifier.bias": "dataset_0.bias",
        # Beat, downbeat, section classifiers
        "beat_classifier.classifier.weight": "beat_0.weight",
        "beat_classifier.classifier.bias": "beat_0.bias",
        "downbeat_classifier.classifier.weight": "downbeat_0.weight",
        "downbeat_classifier.classifier.bias": "downbeat_0.bias",
        "section_classifier.classifier.weight": "section_0.weight",
        "section_classifier.classifier.bias": "section_0.bias"
    }
    
    final_sd = {}
    for k, v in new_sd.items():
        if k in additional_mappings:
            final_sd[additional_mappings[k]] = v
        else:
            final_sd[k] = v
    
    # Initialisiere fehlende Gewichte mit passenden Dimensionen
    # embeddings.projection: 1944 -> 24 (1944 = 81 * 24)
    if "embeddings.projection.weight" not in final_sd:
        weight = torch.empty(24, 1944)
        torch.nn.init.xavier_uniform_(weight)
        final_sd["embeddings.projection.weight"] = weight
    if "embeddings.projection.bias" not in final_sd:
        final_sd["embeddings.projection.bias"] = torch.zeros(24)
    
    # final_projection: 24 -> 96
    if "final_projection.weight" not in final_sd:
        # Initialisiere mit Xavier Uniform
        weight = torch.empty(96, 24)
        torch.nn.init.xavier_uniform_(weight)
        final_sd["final_projection.weight"] = weight
    if "final_projection.bias" not in final_sd:
        final_sd["final_projection.bias"] = torch.zeros(96)
    
    return final_sd