# config.py

from dataclasses import dataclass, field
from typing import List, Union

@dataclass
class Config:
    """Centralized configuration for TTCM experiments (few-shot TAG OOD detection)"""

    # --- Data and dataset paths ---
    data_dir: str = './dataset'
    data_name: str = 'cora'

    # --- Runtime control ---
    gpu: int = 0
    seed: int = 13
    use_llm_cache: bool = True  # Use cached LLM outputs to speed up experiments

    # --- Few-shot settings ---
    n_way: int = 3
    k_shot: int = 5

    # --- TTCM fine-tuning hyperparameters ---
    ft_epochs: int = 50
    ft_lr: float = 1e-5
    ft_batch_size: int = 128
    ft_temp: float = 0.07

    # --- Regularization ---
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1

    # --- DARE engine hyperparameters ---
    dare_lr: float = 1e-3
    dare_hidden: int = 256
    dare_alpha_max: float = 10.0
    dare_m_target_end: float = 0.9  # Kept for logging consistency with argparse
    lambda_s: float = 1.0

    # --- Core DARE hyperparameters (Cora 5-shot reference configuration) ---
    tau_quantile: float = 0.30
    tau_mode: str = 'hybrid'  # 'dual_max', 'dual_avg', 'pos_only', 'target_pos', 'hybrid'
    target_pos: float = 0.75 # for target_pos/hybrid mode
    clamp_min: float = 0.02
    clamp_max: float = 0.8
    r_push_min: float = 0.2
    lambda_clean: float = 0.5
    dare_warmup_epochs: int = 5
    lambda_push: float = 0.2
    lambda_align: float = 0.5
    r_gate: float = 0.0

    # --- Additional: DARE r computation / calibration / prompts / prototype stabilization ---
    dare_r_mode: str = 'margin'  # 'margin' or 'mlp'
    enable_constraint_calibration: bool = True
    use_graph_prompts: bool = True
    fix_ood_prototypes: bool = True
    calibrate_exact: bool = True
    ood_target_rate: float = 0.05
    r_thr_id: float = 0.5
    r_thr_ood: float = 0.1

    # --- Evaluation options ---
    eval_use_energy: bool = False

    # --- Ablation & debug switches ---
    disable_dare_weights: bool = False
    disable_ood_concept: bool = False
    decouple_push: bool = True
    r_oracle_mode: str = 'none'  # 'none', 'hard', 'soft'
    push_selection: str = 'r'  # 'r', 'margin'
    push_q: float = 0.2

    # --- SSDA sample selection hyperparameters (S1: structural, S2: semantic) ---
    # S1 - Structure-based augmentation
    s1_threshold: float = 0.75
    pseudo_k_neighbors: int = 5
    pseudo_k_neighbors_neg: int = 5
    # S2 - Semantic-based augmentation
    s2_threshold: float = 0.35
    see_samples_per_class: int = 10
    see_samples_per_class_neg: int = 10
    s2_k_min: int = 1
    s2_k_max: int = 5

    # --- Hybrid negative mining ---
    hybrid_num_negatives: int = 600
    id_protection_threshold: float = 0.35
    ood_similarity_threshold: float = 0.35

    # --- Static OOD concept retrieval ---
    retrieval_p: int = 32
    llm_candidate_count: int = 200

    # --- Experiment tag (for naming result files) ---
    exp_tag: str = 'default_experiment'

    # --- Model architecture (usually fixed) ---
    gnn_input: int = 1433 # Overwritten by actual feature dimension at runtime
    gnn_hid: int = 768
    gnn_output: int = 768
    embed_dim: int = 768
    context_length: int = 128
    transformer_width: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 12
    vocab_size: int = 49408

    # --- dataloader parameters (not used in S2) ---
    neigh_num: int = 3
    neg_num: int = 5

# Utility helpers for different experiment presets
def get_baseline_config():
    return Config()

def get_optimized_config():
    config = Config()
    config.exp_tag = 'optimized_run'
    config.lambda_push = 0.2
    return config
