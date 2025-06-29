import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class TimeAttentionModule(nn.Module):
    """Multi-Head Attention mit separaten query, key, value Layern."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Container für q, k, v, rpb
        self.self = nn.Module()
        self.self.query = nn.Linear(dim, dim, bias=True)
        self.self.key = nn.Linear(dim, dim, bias=True)
        self.self.value = nn.Linear(dim, dim, bias=True)
        
        # rpb für TimeAttention: [2, 9]
        self.self.rpb = nn.Parameter(torch.randn(2, 9))
        
        # Ausgabe-Projektion
        self.output = nn.Sequential()
        self.output.add_module('dense', nn.Linear(dim, dim, bias=True))
        
    def forward(self, x):
        B, N, C = x.shape
        x_reshaped = x.transpose(0, 1)  # MHA erwartet (Seq, Batch, Feature)

        # Der KORREKTE und FINALE Aufruf. Kein try-except mehr nötig.
        attn_output, _ = F.multi_head_attention_forward(
            query=x_reshaped, 
            key=x_reshaped, 
            value=x_reshaped,
            embed_dim_to_check=self.dim,
            num_heads=self.num_heads,
            in_proj_weight=torch.cat([self.self.query.weight, self.self.key.weight, self.self.value.weight]),
            in_proj_bias=torch.cat([self.self.query.bias, self.self.key.bias, self.self.value.bias]),
            bias_k=None,  # Fehlendes Argument hinzugefügt
            bias_v=None,  # Fehlendes Argument hinzugefügt
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.output.dense.weight,
            out_proj_bias=self.output.dense.bias,
            training=self.training,
            # attn_mask kann hier später für die RPB-Logik verwendet werden, vorerst None
            attn_mask=None 
        )
        return attn_output.transpose(0, 1)  # Zurück zu (Batch, Seq, Feature)



class InstAttentionModule(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Container für q, k, v, rpb
        self.self = nn.Module()
        self.self.query = nn.Linear(dim, dim, bias=True)
        self.self.key = nn.Linear(dim, dim, bias=True)
        self.self.value = nn.Linear(dim, dim, bias=True)
        
        # rpb für InstAttention: [2, 9, 9]
        self.self.rpb = nn.Parameter(torch.randn(2, 9, 9))
        
        # Ausgabe-Projektion
        self.output = nn.Sequential()
        self.output.add_module('dense', nn.Linear(dim, dim, bias=True))
        
    def forward(self, x):
        B, N, C = x.shape
        x_reshaped = x.transpose(0, 1)  # MHA erwartet (Seq, Batch, Feature)

        # Der KORREKTE und FINALE Aufruf. Kein try-except mehr nötig.
        attn_output, _ = F.multi_head_attention_forward(
            query=x_reshaped, 
            key=x_reshaped, 
            value=x_reshaped,
            embed_dim_to_check=self.dim,
            num_heads=self.num_heads,
            in_proj_weight=torch.cat([self.self.query.weight, self.self.key.weight, self.self.value.weight]),
            in_proj_bias=torch.cat([self.self.query.bias, self.self.key.bias, self.self.value.bias]),
            bias_k=None,  # Fehlendes Argument hinzugefügt
            bias_v=None,  # Fehlendes Argument hinzugefügt
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.output.dense.weight,
            out_proj_bias=self.output.dense.bias,
            training=self.training,
            # attn_mask kann hier später für die RPB-Logik verwendet werden, vorerst None
            attn_mask=None 
        )
        return attn_output.transpose(0, 1)  # Zurück zu (Batch, Seq, Feature)



class TimeLayer(nn.Module):
    """Time Layer mit zwei Attention-Schichten."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(dim)
        self.attention = TimeAttentionModule(dim, num_heads)
        # Zweite Attention-Schicht nur für TimeLayer
        self.attention2 = TimeAttentionModule(dim, num_heads)
        self.layernorm_after = nn.LayerNorm(dim)
        
        # MLP mit korrekten Dimensionen: 24 -> 96 -> 24
        self.intermediate = nn.Sequential()
        self.intermediate.add_module('dense', nn.Linear(dim, dim * 4))  # 24 -> 96
        self.intermediate.add_module('activation', nn.GELU())
        
        self.output = nn.Sequential()
        self.output.add_module('dense', nn.Linear(dim * 4, dim))  # 96 -> 24
        
    def forward(self, x):
        # Pre-Norm Architektur: x + attn(norm(x))
        attn_output = self.attention(self.layernorm_before(x))
        x = x + attn_output
        
        # Zweite Attention ohne Pre-Norm
        attn2_output = self.attention2(x)
        x = x + attn2_output
        
        # MLP mit Pre-Norm
        mlp_input = self.layernorm_after(x)
        mlp_intermediate_out = self.intermediate(mlp_input)
        mlp_output = self.output(mlp_intermediate_out)
        x = x + mlp_output
        
        return x

class InstLayer(nn.Module):
    """Inst Layer mit nur einer Attention-Schicht."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(dim)
        self.attention = InstAttentionModule(dim, num_heads)
        # Keine attention2 für InstLayer
        self.layernorm_after = nn.LayerNorm(dim)
        
        # MLP mit korrekten Dimensionen: 24 -> 96 -> 24
        self.intermediate = nn.Sequential()
        self.intermediate.add_module('dense', nn.Linear(dim, dim * 4))  # 24 -> 96
        self.intermediate.add_module('activation', nn.GELU())
        
        self.output = nn.Sequential()
        self.output.add_module('dense', nn.Linear(dim * 4, dim))  # 96 -> 24
        
    def forward(self, x):
        # Pre-Norm Architektur: x + attn(norm(x))
        attn_output = self.attention(self.layernorm_before(x))
        x = x + attn_output
        
        # MLP mit Pre-Norm
        mlp_input = self.layernorm_after(x)
        mlp_intermediate_out = self.intermediate(mlp_input)
        mlp_output = self.output(mlp_intermediate_out)
        x = x + mlp_output
        
        return x



class EmbeddingModule(nn.Module):
    """Eingangs-Einbettung mit exakten Kanalgrößen aus der Blaupause.
    
    Verarbeitet die Eingabe zu einem 24-dimensionalen Merkmalsvektor pro Zeitpunkt.
    """
    def __init__(self, config):
        super().__init__()
        # Exakte Kanalgrößen basierend auf der size mismatch-Fehlermeldung
        self.conv0 = nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=1)  # [B, 1, 9, 12] -> [B, 12, 9, 12]
        self.conv1 = nn.Conv2d(12, 24, kernel_size=(1, 12), stride=1, padding=0)  # [B, 12, 9, 12] -> [B, 24, 9, 1]
        self.conv2 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)  # [B, 24, 9, 1] -> [B, 24, 9, 1]
        
        # Norm auf 24 Dimensionen (letzte Dimension)
        self.norm = nn.LayerNorm(24)  # Norm über die Feature-Dimension
        
    def forward(self, x):
        # x: (batch, stems, freq, time) -> z.B. [B, 4, 81, 1722] für 4 Stems, 81 Frequenzbänder, 1722 Zeitfenster
        batch_size, num_stems, freq_bins, time_steps = x.shape
        
        # Debug-Ausgabe der Eingangsform
        print(f"Embedding input shape: {x.shape} (batch, stems, freq, time)")
        
        # 1. Kombiniere Batch und Stems für parallele Verarbeitung
        x = x.view(-1, 1, freq_bins, time_steps)  # [B*4, 1, 81, 1722]
        print(f"After view: {x.shape} (batch*stems, 1, freq, time)")
        
        # 2. Führe die Faltungen durch
        x = F.gelu(self.conv0(x))  # [B*4, 12, 81, 1722] - Erhöhe die Kanäle auf 12
        print(f"After conv0: {x.shape} (batch*stems, 12, freq, time)")
        
        x = F.gelu(self.conv1(x))  # [B*4, 24, 81, 1] - Reduziere Zeitdimension auf 1, erhöhe Kanäle auf 24
        print(f"After conv1: {x.shape} (batch*stems, 24, freq, 1)")
        
        x = F.gelu(self.conv2(x))  # [B*4, 24, 81, 1] - Behalte Dimensionen bei
        print(f"After conv2: {x.shape} (batch*stems, 24, freq, 1)")
        
        # 3. Verarbeite die 4D-Tensoren korrekt
        print(f"After conv2: {x.shape} (batch*stems, channels, freq, time)")
        
        # Kombiniere Frequenz- und Zeitdimension
        x = x.permute(0, 2, 1, 3)  # [B*4, freq, channels, time]
        x = x.reshape(batch_size * num_stems, -1, x.size(1) * x.size(3))  # [B*4, freq, channels*time]
        print(f"After reshape: {x.shape} (batch*stems, freq, channels*time)")
        
        # Reduziere auf die gewünschte Ausgabedimension
        x = x.mean(dim=-1)  # [B*4, freq]
        print(f"After mean: {x.shape} (batch*stems, freq)")
        
        # Füge eine Dimension für die Channel-Dimension ein
        x = x.unsqueeze(-1)  # [B*4, freq, 1]
        print(f"After unsqueeze: {x.shape} (batch*stems, freq, 1)")
        
        # Projiziere auf 24 Dimensionen
        if not hasattr(self, 'proj'):
            self.proj = nn.Conv1d(x.size(1), 24, kernel_size=1).to(x.device)
        x = self.proj(x)  # [B*4, 24, 1]
        x = x.permute(0, 2, 1)  # [B*4, 1, 24]
        print(f"After projection: {x.shape} (batch*stems, 1, 24)")
        
        # Wiederhole für die Sequenzlänge 9
        if not hasattr(self, 'pos_enc'):
            self.pos_enc = nn.Parameter(torch.randn(1, 9, 24)).to(x.device)
        x = x.expand(-1, 9, -1)  # [B*4, 9, 24]
        x = x + self.pos_enc  # Füge Positionsinformation hinzu
        print(f"After position encoding: {x.shape} (batch*stems, 9, 24)")
        
        # 4. LayerNorm auf die letzte Dimension (24)
        x = self.norm(x)  # [B*4, 9, 24]
        
        # Ausgabe: [B*4, 9, 24] wobei:
        # - B*4: Batch-Größe * Anzahl Stems (z.B. 4 Stems pro Sample)
        # - 9: Sequenzlänge (Zeitschritte)
        # - 24: Feature-Dimension
        return x

class MPSCompatibleAllin1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Eingangs-Einbettung (Output: [B, 9, 24])
        self.embedding = EmbeddingModule(config)  # Output: [B, 9, 24]
        
        # 2. Encoder-Schichten (24-Dimensionen durchgehend)
        # Handle both dict and object-style config
        num_attention_heads = config.get('num_attention_heads', 8) if isinstance(config, dict) else config.num_attention_heads
        num_hidden_layers = config.get('num_hidden_layers', 12) if isinstance(config, dict) else config.num_hidden_layers
        
        self.encoder_layers = nn.ModuleList([
            nn.ModuleList([
                TimeLayer(dim=24, num_heads=num_attention_heads),  # 24-Dimensionen
                InstLayer(dim=24, num_heads=num_attention_heads)   # 24-Dimensionen
            ]) for _ in range(num_hidden_layers)
        ])
        
        # 3. Finale Projektion von 24 auf 96 Dimensionen
        self.final_projection = nn.Linear(24, 96)
        
        # 4. Finale LayerNorm auf 96 Dimensionen
        self.final_layer_norm = nn.LayerNorm(96)
        
        # 5. Klassifikatoren (alle erwarten 96-Dimensionen)
        # Funktionen-Klassifikatoren
        self.function_classifier = nn.ModuleDict({
            'harmonix': nn.ModuleDict({
                'classifier': nn.Linear(96, 10)  # 10 Funktionen für Harmonix
            }),
            'raveform': nn.ModuleDict({
                'classifier': nn.Linear(96, 11)  # 11 Techniken für RaveForm
            })
        })
        
        # Weitere Klassifikatoren
        self.beat_classifier = nn.ModuleDict({
            'classifier': nn.Linear(96, 1)  # Beat-Vorhersage
        })
        
        self.downbeat_classifier = nn.ModuleDict({
            'classifier': nn.Linear(96, 1)  # Downbeat-Vorhersage
        })
        
        self.section_classifier = nn.ModuleDict({
            'classifier': nn.Linear(96, 1)  # Abschnitts-Vorhersage
        })
        
        # Dataset-Klassifikator
        self.dataset_classifier = nn.Linear(96, 2)  # 2 Datensätze
        
        # Technik-Klassifikator (falls benötigt)
        self.technique_classifier = nn.Linear(96, 11)  # 11 Techniken
        
        # 6. Dataset Classifier
        self.dataset_classifier = nn.Linear(96, 2)  # 2 basierend auf bias shape
        
        # Initialisierung
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # 1. Embedding -> Output ist 24-dim [B, 9, 24]
        x = self.embedding(x)  # (batch*stems, seq_len, 24)
        
        # 2. Encoder -> Arbeitet durchgehend mit 24-dim
        for time_layer, inst_layer in self.encoder_layers:
            x = time_layer(x)  # TimeLayer verarbeitet [B, 9, 24] -> [B, 9, 24]
            x = inst_layer(x)  # InstLayer verarbeitet [B, 9, 24] -> [B, 9, 24]
        
        # 3. Finale Projektion auf 96-dim
        x = self.final_projection(x)  # [B, 9, 96]
        
        # 4. Finale LayerNorm auf 96-dim
        x = self.final_layer_norm(x)  # [B, 9, 96]
        
        # 5. Mitteln über die Sequenzlänge für globale Merkmale
        x = x.mean(dim=1)  # [B, 96]
        
        # 6. Berechne die tatsächliche Batch-Größe (Anzahl der Stems)
        # Annahme: Die ursprüngliche Batch-Größe ist 1/4 der aktuellen Größe, da wir 4 Stems pro Sample haben
        batch_size = x.size(0) // 4  # Anzahl der Stems
        
        # 7. Berechne die Logits für jede Aufgabe
        # Beat, Downbeat und Segment Klassifikation
        beat_logits = self.beat_classifier['classifier'](x).view(batch_size, 4)  # [batch, 4]
        downbeat_logits = self.downbeat_classifier['classifier'](x).view(batch_size, 4)  # [batch, 4]
        segment_logits = self.section_classifier['classifier'](x).view(batch_size, 4)  # [batch, 4]
        
        # Funktionale Klassifikation (für jeden Datensatz)
        functional_harmonix = self.function_classifier['harmonix']['classifier'](x).view(batch_size, 4, -1)  # [batch, 4, 10]
        functional_raveform = self.function_classifier['raveform']['classifier'](x).view(batch_size, 4, -1)  # [batch, 4, 11]
        
        # Dataset-Klassifikation
        dataset_logits = self.dataset_classifier(x)  # [B, 2]
        
        # Technik-Klassifikation (falls benötigt)
        technique_logits = self.technique_classifier(x)  # [B, 11]
        
        return {
            'beat': beat_logits,  # [batch, 4]
            'downbeat': downbeat_logits,  # [batch, 4]
            'segment': segment_logits,  # [batch, 4]
            'functional': {
                'harmonix': functional_harmonix,  # [batch, 4, 10]
                'raveform': functional_raveform  # [batch, 4, 11]
            },
            'dataset': dataset_logits,  # [B, 2]
            'technique': technique_logits  # [B, 11]
        }
