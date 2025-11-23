import torch
import torch.nn as nn
from mamba_ssm import Mamba

class GatedFusion(nn.Module):
    """
    Learns to dynamically weight the importance of Clinical Notes vs. ICD Codes.
    Formula: h_fused = gamma * Note + (1 - gamma) * Code
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Input: Concatenation of (Code_Dim + Note_Dim) -> Output: 1 scalar (Gate)
        self.gate_net = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, struct_embeds, note_embeds):
        # struct_embeds: [Batch, Seq_Len, Dim]
        # note_embeds:   [Batch, 1, Dim] -> Need to expand to match Seq_Len
        
        seq_len = struct_embeds.size(1)
        note_expanded = note_embeds.expand(-1, seq_len, -1)
        
        # Concatenate along feature dimension
        combined = torch.cat([struct_embeds, note_expanded], dim=-1)
        
        # Calculate Gate (Gamma)
        gamma = self.sigmoid(self.gate_net(combined))
        
        # Weighted Sum
        return gamma * note_expanded + (1 - gamma) * struct_embeds

class MambaEHR(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Project 768-dim BERT vectors down to model dimension (256)
        self.note_projector = nn.Linear(768, d_model)
        
        # Fusion Layer
        self.fusion = GatedFusion(d_model)
        
        # Mamba Backbone
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.binary_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        
    def forward(self, input_ids, note_tensor=None, task = 'forecasting'):
        # 1. Embed ICD Codes
        x = self.embedding(input_ids) # [Batch, Seq, Dim]
        
        # 2. Multimodal Fusion (Optional)
        if note_tensor is not None:
            # Project BERT vector to 256 dims
            note_proj = self.note_projector(note_tensor) # [Batch, Dim]
            
            # Add Sequence Dimension for broadcasting: [Batch, 1, Dim]
            if note_proj.dim() == 2:
                note_proj = note_proj.unsqueeze(1)
                
            # Fuse
            x = self.fusion(x, note_proj)
            
        # 3. Mamba Layers
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        
        if task == 'mortality':
            x_pool = x.max(dim=1)[0]
            return self.binary_head(x_pool)
        else:
            return self.lm_head(x)