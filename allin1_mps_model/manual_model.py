import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class DilatedNeighborhoodAttention1D(nn.Module):
    """
    1D Dilated Neighborhood Attention als Ersatz für NATTEN.
    Verwendet Standard-PyTorch-Funktionen für MPS-Kompatibilität.
    """
    def __init__(self, dim: int, num_heads: int, dilation: int, window_size: int = 7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dilation = dilation
        self.window_size = window_size
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Relative Positionsbias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
        # Erzeuge Lookup-Tabelle für relative Positionen
        coords = torch.arange(window_size)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords += window_size - 1
        self.register_buffer("relative_position_index", relative_coords)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Lineare Projektionen
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Skalierter Dot-Product Attention mit Fensterbildung
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        # Relative Positionsbias hinzufügen (nur wenn die Fenstergröße kleiner oder gleich der Sequenzlänge ist)
        seq_len = x.size(1)
        if self.window_size <= seq_len:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size, self.window_size, -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            
            # Stelle sicher, dass die Bias-Dimensionen mit den Attention-Dimensionen übereinstimmen
            if seq_len <= self.window_size:
                relative_position_bias = relative_position_bias[:, :seq_len, :seq_len]
            else:
                # Wenn die Sequenz länger ist als das Fenster, wiederhole die Bias-Matrix
                repeats = (seq_len + self.window_size - 1) // self.window_size
                relative_position_bias = relative_position_bias.repeat(1, repeats, repeats)
                relative_position_bias = relative_position_bias[:, :seq_len, :seq_len]
            
            attn = attn + relative_position_bias.unsqueeze(0)
        else:
            # Wenn das Fenster größer ist als die Sequenz, verwende nur die ersten seq_len x seq_len Einträge
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ][:seq_len*seq_len].view(
                seq_len, seq_len, -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        
        # Softmax über die Aufmerksamkeitsgewichte
        attn = F.softmax(attn, dim=-1)
        
        # Gewichtete Summe der Werte
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MPSCompatibleAllin1(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Extrahiere die Architektur-Parameter
        self.d_model = config['hidden_size']
        self.n_head = config['num_attention_heads']
        self.n_layers = config['num_hidden_layers']
        self.dim_feedforward = config['intermediate_size']
        self.num_labels = config['num_labels']
        
        # 1. Frequenz- und Zeit-Embedding
        # Eingabe: (batch_size, num_stems=4, freq_bands=81, time_frames)
        self.freq_embed = nn.Linear(81, self.d_model)  # Projiziere 81 Frequenzbänder in d_model Dimensionen
        
        # 2. Positionelles Encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=0.1)
        
        # 2.5 Stem Embedding
        self.stem_embed = nn.Embedding(4, self.d_model)  # 4 Stems: bass, drums, other, vocals
        
        # 3. Transformer Encoder Layer mit Dilated Attention
        encoder_layers = []
        for i in range(self.n_layers):
            encoder_layers.append(
                TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.n_head,
                    dim_feedforward=self.dim_feedforward,
                    dilation=2**i,  # Exponentiell wachsende Dilatation
                    dropout=0.1
                )
            )
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 4. Ausgabeschichten für die verschiedenen Aufgaben
        self.beat_head = nn.Linear(self.d_model, 1)
        self.downbeat_head = nn.Linear(self.d_model, 1)
        self.segment_head = nn.Linear(self.d_model, 1)
        self.functional_head = nn.Linear(self.d_model, self.num_labels)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> dict:
        # ANFANGSFORM: (batch, stems, freq, time)
        # z.B. (1, 4, 81, 11015)
        print(f"Input shape: {x.shape}")
        batch_size, num_stems, num_freq, num_timesteps = x.shape
        
        # === SCHRITT 1: Frequenz-Embedding ===
        # (batch, stems, freq, time) -> (batch * stems * time, freq)
        x = x.permute(0, 1, 3, 2)  # (batch, stems, time, freq)
        print(f"After permute: {x.shape}")
        
        # Reshape für lineare Schicht
        x = x.reshape(-1, num_freq)  # (batch * stems * time, freq)
        print(f"Reshaped for linear: {x.shape}")
        
        # Frequenz-Embedding anwenden
        x = self.freq_embed(x)  # (batch * stems * time, d_model)
        print(f"After freq_embed: {x.shape}")
        
        # Zurück zur 3D-Form für den Transformer
        # (batch * stems, time, d_model)
        x = x.view(batch_size * num_stems, num_timesteps, -1)
        print(f"Reshaped for transformer: {x.shape}")
        
        # === SCHRITT 2: Positionelles Encoding hinzufügen ===
        x = self.pos_encoder(x)  # (batch * stems, time, d_model)
        print(f"After pos_encoder: {x.shape}")
        
        # === SCHRITT 3: Stem-Embedding hinzufügen ===
        stem_ids = torch.arange(num_stems, device=x.device).repeat(batch_size)  # (batch*stems,)
        stem_emb = self.stem_embed(stem_ids)  # (batch*stems, d_model)
        x = x + stem_emb.unsqueeze(1)  # Add to each time step
        print(f"After stem_embed: {x.shape}")
        
        # === SCHRITT 4: Transformer-Encoder ===
        x = self.encoder(x)  # (batch*stems, time, d_model)
        print(f"After encoder: {x.shape}")
        
        # === SCHRITT 5: Aufgaben-spezifische Ausgänge ===
        # Durchschnitt über die Zeitachse
        x_mean = x.mean(dim=1)  # (batch*stems, d_model)
        print(f"After mean: {x_mean.shape}")
        
        # Separate Heads für jeden Aufgabentyp
        beat_logits = self.beat_head(x_mean)  # (batch*stems, 1)
        downbeat_logits = self.downbeat_head(x_mean)  # (batch*stems, 1)
        segment_logits = self.segment_head(x_mean)  # (batch*stems, 1)
        functional_logits = self.functional_head(x_mean)  # (batch*stems, num_labels)
        
        # Squeeze die letzte Dimension für die binären Klassifikatoren
        beat_logits = beat_logits.squeeze(-1)  # (batch*stems,)
        downbeat_logits = downbeat_logits.squeeze(-1)
        segment_logits = segment_logits.squeeze(-1)
        
        # Zurück zur (batch, stems, ...) Form
        beat_logits = beat_logits.view(batch_size, num_stems)
        downbeat_logits = downbeat_logits.view(batch_size, num_stems)
        segment_logits = segment_logits.view(batch_size, num_stems)
        functional_logits = functional_logits.view(batch_size, num_stems, -1)
        
        return {
            'beat': beat_logits,
            'downbeat': downbeat_logits,
            'segment': segment_logits,
            'functional': functional_logits
        }
        x = x.reshape(batch_size, num_stems, time_steps, -1)   # (batch, 4, time_steps, d_model)
        
        # 4. Aufgaben-spezifische Köpfe
        # Durchschnitt über die Stem-Dimension
        x_mean = x.mean(dim=1)  # (batch, time_steps, d_model)
        
        outputs = {
            'beat': self.beat_head(x_mean).squeeze(-1),
            'downbeat': self.downbeat_head(x_mean).squeeze(-1),
            'segment': self.segment_head(x_mean).squeeze(-1),
            'functional': self.functional_head(x_mean)
        }
        
        return outputs

class PositionalEncoding(nn.Module):
    """Implementiert das sinusförmige Positions-Encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    """Ein einzelner Transformer-Encoder-Layer mit Dilated Attention."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, dilation: int = 1):
        super().__init__()
        self.self_attn = DilatedNeighborhoodAttention1D(d_model, nhead, dilation)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu
        
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # Self-Attention
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-Forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src
