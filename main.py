import os
import torch
import numpy as np
import argparse

# Import core TTCM modules
from src.models.encoders import GraphTextEncoder
from src.models.dare import AnchorAwareDARE
from src.data.data_module import (
    load_data_from_pt,
    preprocess_data_for_fewshot,
    create_fixed_graph_split,
    prepare_training_data
)
from src.train.trainer import train_concurrently
from src.utils.results import ExperimentLogger
from src.utils.common import setup_seed, calculate_metrics
from src.eval.evaluator import evaluate_model
from src.config import Config
from torch import optim

def _namespace_to_config(ns: argparse.Namespace) -> Config:
    cfg = Config()
    ns_dict = vars(ns)
    for k, v in ns_dict.items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    if ns_dict.get('exp_tag') is not None:
        cfg.exp_tag = ns_dict['exp_tag']
    return cfg

def main(cli_args):
    args = _namespace_to_config(cli_args)
    args.id_classes = getattr(cli_args, 'id_classes', None)
    setup_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # --- 1. Data loading and graph split (TAG few-shot setting) ---
    dataset_pt_path = os.path.join(args.data_dir, args.data_name, f'{args.data_name}_processed.pt')
    data_obj = load_data_from_pt(device, args.data_name, dataset_pt_path)

    if not hasattr(data_obj, 'raw_text'):
        if hasattr(data_obj, 'clean_text'):
            data_obj.raw_text = data_obj.clean_text
        elif hasattr(data_obj, 'raw_texts'):
            data_obj.raw_text = data_obj.raw_texts

    class_to_indices_map = preprocess_data_for_fewshot(data_obj)

    import json as _json
    map_txt = os.path.join(args.data_dir, args.data_name, 'label_names.txt')
    map_json = os.path.join(args.data_dir, args.data_name, 'label_names.json')
    try:
        if os.path.exists(map_json):
            with open(map_json, 'r', encoding='utf-8') as f:
                _names = _json.load(f)
            if isinstance(_names, list) and len(_names) == len(class_to_indices_map):
                data_obj.label_names = _names
        elif os.path.exists(map_txt):
            with open(map_txt, 'r', encoding='utf-8') as f:
                _names = [line.strip() for line in f if line.strip()]
            if len(_names) == len(class_to_indices_map):
                data_obj.label_names = _names
    except Exception:
        pass

    if args.data_name.lower() == 'citeseer' and (not hasattr(data_obj, 'label_names') or not data_obj.label_names):
        data_obj.label_names = ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI']
    elif args.data_name.lower() == 'history' and (not hasattr(data_obj, 'label_names') or not data_obj.label_names):
        data_obj.label_names = [f'history_class_{i}' for i in range(len(class_to_indices_map))]

    if hasattr(data_obj, 'label_names') and data_obj.label_names:
        all_class_names = data_obj.label_names
    elif hasattr(data_obj, 'label_text') and data_obj.label_text:
        all_class_names = data_obj.label_text
    else:
        all_class_names = [f'class_{i}' for i in range(len(class_to_indices_map))]

    id_indices_arg = getattr(args, 'id_classes', None)
    if id_indices_arg:
        try:
            id_class_indices = [int(x) for x in id_indices_arg.split(',') if x.strip()!='']
            if len(id_class_indices) != args.n_way:
                id_class_indices = list(range(args.n_way))
        except Exception:
            id_class_indices = list(range(args.n_way))
    else:
        id_class_indices = list(range(args.n_way))

    try:
        id_class_names_to_use = [all_class_names[i] for i in id_class_indices]
    except Exception:
        id_class_indices = list(range(args.n_way))
        id_class_names_to_use = all_class_names[:args.n_way]

    split_data = create_fixed_graph_split(
        class_to_indices_map, 
        id_class_indices, 
        args.k_shot, 
        0.8, 
        args.seed, 
        edge_index=data_obj.edge_index
    )
    
    support_indices_k_shot, support_labels_k_shot = split_data["support_indices"], split_data["support_labels"]
    train_indices_pool = split_data["train_indices_pool"] 
    test_indices, test_labels_original = split_data["test_indices"], split_data["test_multiclass_labels"] 
    class_mapping = split_data["class_mapping"]
    
    # --- 2. Graph-Text dual encoder + pre-trained weights ---
    expected_dim = data_obj.x.shape[1]
    if getattr(args, 'gnn_input', None) is None or args.gnn_input != expected_dim:
        args.gnn_input = expected_dim
    
    model = GraphTextEncoder(args, setup_optimizer=False, raw_feature_dim=None).to(device)
    pretrain_model_frozen = GraphTextEncoder(args, setup_optimizer=False, raw_feature_dim=None).to(device)
    dare_engine = AnchorAwareDARE(args.embed_dim, args.dare_hidden).to(device)
    
    try:
        res_dir_name = 'arxiv' if args.data_name == 'ogbn_arxiv' else args.data_name
        gcap_pretrained_path = os.path.join('res', res_dir_name, 'pretrained_model.pkl')
        if not os.path.exists(gcap_pretrained_path):
            print(f"Warning: Pre-trained model not found at {gcap_pretrained_path}. Attempting to generate it now...")
            import subprocess
            prepare_script_path = os.path.join(os.path.dirname(__file__), 'src', 'pretrain', 'prepare_datasets.py')
            command = [
                'python',
                prepare_script_path,
                '--data_name', args.data_name,
                '--gpu', str(args.gpu)
            ]
            try:
                subprocess.run(command, check=True, text=True)
                print("Pre-computation finished successfully. Resuming main task...")
            except subprocess.CalledProcessError as e:
                print(f"Error: Automated pre-computation failed. Please run the script manually: {' '.join(command)}")
                return
             
        raw_state = torch.load(gcap_pretrained_path, map_location=device)
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in raw_state.items() if k in model_state and model_state[k].shape == v.shape}
        
        model.load_state_dict(filtered_state, strict=False)
        pretrain_model_frozen.load_state_dict(filtered_state, strict=False)
        print(f"Successfully loaded pre-trained weights from {gcap_pretrained_path}")
    except Exception as e:
        print(f"Error loading pre-trained model: {e}"); return
        
    for param in pretrain_model_frozen.parameters():
        param.requires_grad = False
    
    # --- 3-5. SSDA data preparation: OOD concept retrieval + S1/S2 loaders ---
    retrieved_ood_words, dare_loader, neg_loader, clean_loader = prepare_training_data(
        pretrain_model_frozen,
        data_obj,
        id_class_names_to_use,
        class_mapping,
        args, device,
        support_indices_k_shot,
        support_labels_k_shot,
        train_indices_pool,
    )

    # --- 6. Optimizers (CGEM: GNN L1 frozen for stability) ---
    main_params = []
    for name, param in model.named_parameters():
        if "gnn." in name:
            if "vars.0" in name or "vars.1" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                main_params.append(param)
        elif "transformer." in name or "text_projection" in name or \
             "token_embedding" in name or "positional_embedding" in name or \
             "ln_final" in name:
            param.requires_grad = True
            main_params.append(param)
        else:
            param.requires_grad = False 
            
    optimizer_main = optim.Adam(main_params, lr=args.ft_lr, weight_decay=args.weight_decay)
    optimizer_dare = optim.Adam(dare_engine.parameters(), lr=args.dare_lr) 
    
    # --- 7. CGEM training: concurrent optimization (E-step: DARE, M-step: encoders) ---
    train_concurrently(
        model, 
        dare_engine, 
        optimizer_main, optimizer_dare, 
        data_obj, 
        id_class_names_to_use, retrieved_ood_words,
        args, device,
        class_mapping,
        clean_loader,
        neg_loader,
        dare_loader,
        support_indices_k_shot,
        support_labels_k_shot
    )
    
    # --- 8. Final evaluation ---
    test_preds_original, final_ood_scores = evaluate_model(
        model,
        test_indices, 
        data_obj.x, data_obj.edge_index,
        id_class_names_to_use,
        retrieved_ood_words, 
        class_mapping,
        args, device,
        test_labels_original=test_labels_original
    )
    
    if final_ood_scores is None:
        metrics_dict = {
            'AUROC': 0.0, 'AUPR': 0.0, 'FPR95': 1.0,
            'IND_Accuracy': 0.0, 'IND_Macro_F1': 0.0
        }
    else:
        binary_ood_labels = np.array([0 if label in id_class_indices else 1 for label in test_labels_original])
        id_mask = (binary_ood_labels == 0)
        id_true_labels = test_labels_original[id_mask]
        id_preds = test_preds_original[id_mask]
        
        auroc, aupr, fpr95, ind_acc, ind_f1 = calculate_metrics(
            binary_ood_labels, 
            final_ood_scores,
            id_preds, 
            id_true_labels
        )
        
        print("\n" + f"--- TTCM Final Results ({args.data_name}, {args.k_shot}-shot, seed={args.seed}) ---")
        print(f"{'AUROC'.ljust(8)}: {auroc:.2%}")
        print(f"{'AUPR'.ljust(8)}: {aupr:.2%}")
        print(f"{'FPR95'.ljust(8)}: {fpr95:.2%}")
        print(f"{'IND_ACC'.ljust(8)}: {ind_acc:.2%}")
        print(f"{'IND_F1'.ljust(8)}: {ind_f1:.2%}")
        print("---------------------------------")
        
        metrics_dict = {
            'AUROC': auroc,
            'AUPR': aupr,
            'FPR95': fpr95,
            'IND_Accuracy': ind_acc,
            'IND_Macro_F1': ind_f1
        }

    # --- Save summary results (for experiments_TTCM.{csv,jsonl}) ---
    logger = ExperimentLogger(args)
    mode = 'wodare' if getattr(args, 'disable_dare_weights', False) else 'full'
    logger.save_metrics(metrics_dict, mode=mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TTCM (DARE, concurrent) Experiment')
    
    # --- Paths and Control ---
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--data_name', type=str, default='cora')
    parser.add_argument('--exp_tag', type=str, default=None, help='Experiment tag for result filenames')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--use_llm_cache', action='store_true', help="Use cached LLM OOD words")
    
    # --- Few-Shot Settings ---
    parser.add_argument('--n_way', type=int, default=3, help="Number of ID classes")
    parser.add_argument('--k_shot', type=int, default=2, help="Number of k-shot samples per ID class")
    parser.add_argument('--id_classes', type=str, default=None, help='Manually specify ID class indices, comma-separated (e.g., 0,3,4)')
    
    # --- Fine-tuning Hyperparameters ---
    parser.add_argument('--ft_epochs', type=int, default=50)
    parser.add_argument('--ft_lr', type=float, default=1e-5)
    parser.add_argument('--ft_batch_size', type=int, default=128)
    parser.add_argument('--ft_temp', type=float, default=0.07)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # --- DARE Hyperparameters ---
    parser.add_argument('--dare_lr', type=float, default=1e-3)
    parser.add_argument('--dare_hidden', type=int, default=256)
    parser.add_argument('--dare_alpha_max', type=float, default=10.0)
    parser.add_argument('--lambda_s', type=float, default=1.0)
    parser.add_argument('--tau_quantile', type=float, default=0.1)
    parser.add_argument('--clamp_min', type=float, default=0.1)
    parser.add_argument('--lambda_clean', type=float, default=0.5)
    parser.add_argument('--lambda_push', type=float, default=0.1)
    parser.add_argument('--lambda_align', type=float, default=0.5)
    
    # --- Ablation Switches ---
    parser.add_argument('--disable_dare_weights', action='store_true', help="Disable DARE weights (r=1.0) for ablation")
    parser.add_argument('--disable_ood_concept', action='store_true', help='Disable OOD concepts for ablation')
    parser.add_argument('--decouple_push', action='store_true', help='Decouple U-set loss (CE with r, Push with 1-r)')
        
    # --- SSDA Hyperparameters ---
    parser.add_argument('--see_samples_per_class', type=int, default=10, help="[S2] Target Top-K samples per ID class")
    parser.add_argument('--s2_threshold', type=float, default=0.3, help="[S2] Confidence threshold for semantic augmentation")
    parser.add_argument('--s2_k_min', type=int, default=1)
    parser.add_argument('--s2_k_max', type=int, default=5)
    parser.add_argument('--see_samples_per_class_neg', type=int, default=10, help="[S2] Target Bottom-K samples per ID class")
    parser.add_argument('--pseudo_k_neighbors', type=int, default=5, help="[S1] K neighbors for structural pseudo-labels")
    parser.add_argument('--s1_threshold', type=float, default=0.7, help="[S1] Confidence threshold for structural augmentation")
    parser.add_argument('--pseudo_k_neighbors_neg', type=int, default=5, help="[S1] K neighbors for negative samples")
        
    # --- Static OOD Retrieval ---
    parser.add_argument('--retrieval_p', type=int, default=32, help="Size of the static balanced OOD pool")
    parser.add_argument('--llm_candidate_count', type=int, default=200, help="Total candidates from LLM")
        
    # --- Model Architecture ---
    sbert_dim = 768
    parser.add_argument('--gnn_hid', type=int, default=sbert_dim)
    parser.add_argument('--gnn_output', type=int, default=sbert_dim)
    parser.add_argument('--embed_dim', type=int, default=sbert_dim)
    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12) 
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)
    
    args = parser.parse_args()

    main(args)
