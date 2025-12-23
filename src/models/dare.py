# dare.py: DARE family models (standalone from encoders, used in TTCM/CGEM)
from __future__ import annotations

import torch
from torch import nn

__all__ = ['AnchorAwareDARE', 'CA_DARE', 'DistributionAwareReweightingEngine']


class DistributionAwareReweightingEngine(nn.Module):
    """
    Distribution-Aware Reweighting Engine (DARE)
    """
    def __init__(self, emb_dim, hidden=256):
        super().__init__()
        self.alpha_g = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.alpha_t = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + 4, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1)
        )

    def forward(self, g_emb, t_emb, scalar_feats):
        g_w = g_emb * self.alpha_g
        t_w = t_emb * self.alpha_t
        multi_modal = torch.cat([g_w, t_w, scalar_feats], dim=-1)
        s = self.mlp(multi_modal).squeeze(-1)
        return s


class AnchorAwareDARE(nn.Module):
    """
    Anchor-Aware DARE: explicit reasoning on three criteria (s_align, s_sem, s_struct)
    """
    def __init__(self, emb_dim, hidden=64):
        super().__init__()
        input_dim = 3
        self.decision_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1)
        )

    def forward(self, g_emb_norm, t_emb_norm, id_text_protos_norm, id_graph_protos_norm):
        s_align = (g_emb_norm * t_emb_norm).sum(dim=-1, keepdim=True)
        sim_sem_all = t_emb_norm @ id_text_protos_norm.t()
        s_sem, _ = torch.max(sim_sem_all, dim=1, keepdim=True)
        sim_struct_all = g_emb_norm @ id_graph_protos_norm.t()
        s_struct, _ = torch.max(sim_struct_all, dim=1, keepdim=True)
        explicit_features = torch.cat([s_align, s_sem, s_struct], dim=-1)
        r_logits = self.decision_mlp(explicit_features).squeeze(-1)
        return r_logits


class CA_DARE(nn.Module):
    """
    Confidence-Aware DARE (CA-DARE)
    """
    def __init__(self, emb_dim, hidden=256, confidence_gate=0.7):
        super().__init__()
        self.mismatch_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1)
        )
        self.confidence_gate = confidence_gate

    def forward(self, g_emb, t_emb, id_confidence=None):
        multi_modal_emb = torch.cat([g_emb, t_emb], dim=-1)
        s_mismatch = self.mismatch_mlp(multi_modal_emb).squeeze(-1)
        if id_confidence is not None:
            confidence_factor = torch.sigmoid((id_confidence - self.confidence_gate) * 10.0)
            protection_bonus = confidence_factor * 2.0
            s_modulated = s_mismatch + protection_bonus
            return s_modulated
        else:
            return s_mismatch
