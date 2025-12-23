from __future__ import annotations
import os
import re
import json
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm
import src.models.encoders as core_models
# ------------------------------
# Data loading utilities have been moved to data_module; re-import here
# to keep interfaces backward compatible.
# ------------------------------
from src.data.data_module import (
    load_data_from_pt,
    preprocess_data_for_fewshot,
    create_fixed_graph_split,
    DataHelper,
    InBatchDataset,
)

# Explicit bindings to avoid missing local names
tokenize = core_models.tokenize
GraphTextEncoder = core_models.GraphTextEncoder

# Optional LLM utilities
try:
    from src.utils.llm_utils import call_llm_api
except Exception:
    def call_llm_api(messages, debug=False):
        return ""

# =========================
# S1/S2 sample selection (merged from legacy data.py)
# =========================
@torch.no_grad()
def get_structure_based_augmented_set(pretrain_model, all_node_features, all_edge_index, support_indices, support_labels, train_indices, n_way, k_shot, data_obj, class_mapping, args, device, quiet: bool = True):
    if hasattr(args, 'ablation_mode') and args.ablation_mode == 'no_s1':
        return None, None, None, None, None, {}
    summary = {'s1_top_noise': 0, 's1_top_id_correct': 0, 's1_top_id_total': 0, 's1_top_total': 0, 's1_bot_total': 0, 's1_k_shot_total': len(support_indices)}
    top_k_per_class = k_shot * args.pseudo_k_neighbors
    bot_k_per_class = k_shot * args.pseudo_k_neighbors_neg
    if top_k_per_class == 0 and bot_k_per_class == 0:
        return None, None, None, None, None, summary
    pretrain_model.eval()
    support_feats = pretrain_model.encode_image(torch.tensor(support_indices, dtype=torch.long).to(device), all_node_features, all_edge_index)
    if len(train_indices) == 0:
        return None, None, None, None, None, summary
    train_feats = pretrain_model.encode_image(torch.tensor(train_indices, dtype=torch.long).to(device), all_node_features, all_edge_index)
    support_feats = F.normalize(support_feats, p=2, dim=-1)
    train_feats = F.normalize(train_feats, p=2, dim=-1)
    sim_matrix = train_feats @ support_feats.t()
    top_aug_indices_list, top_aug_labels_list, top_aug_true_labels_list = [], [], []
    bot_aug_indices_list, bot_aug_true_labels_list = [], []
    all_true_labels = data_obj.y.cpu().numpy()
    support_labels_tensor = torch.tensor(support_labels, dtype=torch.long).to(device)
    for class_id in range(n_way):
        support_indices_for_class = (support_labels_tensor == class_id).nonzero().squeeze(-1)
        if len(support_indices_for_class) == 0: continue
        sim_matrix_for_class = sim_matrix[:, support_indices_for_class]
        confidence_for_class, _ = torch.max(sim_matrix_for_class, dim=1)
        if top_k_per_class > 0:
            top_k_conf, top_k_indices_in_train = torch.topk(confidence_for_class, k=min(top_k_per_class, len(confidence_for_class)))
            threshold_mask = top_k_conf > args.s1_threshold
            final_indices_in_train = top_k_indices_in_train[threshold_mask]
            selected_train_node_ids = train_indices[final_indices_in_train.cpu().numpy()]
            top_aug_indices_list.append(selected_train_node_ids)
            top_aug_labels_list.append(np.full(len(selected_train_node_ids), class_id))
            top_aug_true_labels_list.append(all_true_labels[selected_train_node_ids])
        if bot_k_per_class > 0:
            bot_k_conf, bot_k_indices_in_train = torch.topk(confidence_for_class, k=min(bot_k_per_class, len(confidence_for_class)), largest=False)
            selected_train_node_ids_neg = train_indices[bot_k_indices_in_train.cpu().numpy()]
            bot_aug_indices_list.append(selected_train_node_ids_neg)
            bot_aug_true_labels_list.append(all_true_labels[selected_train_node_ids_neg])
    if top_aug_indices_list:
        final_top_indices = np.concatenate(top_aug_indices_list)
        final_top_labels = np.concatenate(top_aug_labels_list)
        final_top_true_labels = np.concatenate(top_aug_true_labels_list)
        labels_mapped = [class_mapping.get(l, -1) for l in final_top_true_labels]
        labels_mapped_np = np.array(labels_mapped)
        id_mask = (labels_mapped_np != -1)
        summary['s1_top_noise'] = np.sum(labels_mapped_np == -1)
        summary['s1_top_id_total'] = np.sum(id_mask)
        summary['s1_top_total'] = len(final_top_indices)
        if summary['s1_top_id_total'] > 0:
            correct_predictions = (final_top_labels[id_mask] == labels_mapped_np[id_mask])
            summary['s1_top_id_correct'] = np.sum(correct_predictions)
    else:
        final_top_indices = final_top_labels = final_top_true_labels = None
    if bot_aug_indices_list:
        final_bot_indices = np.concatenate(bot_aug_indices_list)
        final_bot_true_labels = np.concatenate(bot_aug_true_labels_list)
        summary['s1_bot_total'] = len(final_bot_indices)
    else:
        final_bot_indices = final_bot_true_labels = None
    return final_top_indices, final_top_labels, final_top_true_labels, final_bot_indices, final_bot_true_labels, summary

@torch.no_grad()
def get_semantic_based_augmented_set(pretrain_model_frozen, data_obj, train_indices, support_indices, support_labels, id_class_names, class_mapping, args, device, quiet: bool = True):
    if hasattr(args, 'ablation_mode') and args.ablation_mode == 'no_s2':
        return None, None, None, None, None, {}
    summary = {'s2_top_noise': 0, 's2_top_id_correct': 0, 's2_top_id_total': 0, 's2_top_total': 0, 's2_bot_total': 0}
    top_k_per_class = args.see_samples_per_class
    bot_k_per_class = args.see_samples_per_class_neg
    if top_k_per_class == 0 and bot_k_per_class == 0:
        return None, None, None, None, None, summary
    if train_indices is None or len(train_indices) == 0:
        return None, None, None, None, None, summary
    pretrain_model_frozen.eval()
    n_way = len(id_class_names)
    name_feats_list = []
    for class_name in id_class_names:
        prompt = f"a photo of a {class_name.replace('_', ' ')}"
        tokens = tokenize([prompt], context_length=args.context_length).to(device)
        feats = F.normalize(pretrain_model_frozen.encode_text(tokens), p=2, dim=-1)
        name_feats_list.append(feats)
    P_name = torch.cat(name_feats_list, dim=0)
    anchor_texts = [str(data_obj.raw_text[idx]) for idx in support_indices]
    anchor_labels_tensor = torch.tensor(support_labels, dtype=torch.long).to(device)
    batch_size = 512
    anchor_feats_list = []
    for i in range(0, len(anchor_texts), batch_size):
        batch = anchor_texts[i:i+batch_size]
        tokens = tokenize(batch, context_length=args.context_length).to(device)
        feats = F.normalize(pretrain_model_frozen.encode_text(tokens), p=2, dim=-1)
        anchor_feats_list.append(feats)
    anchor_feats = torch.cat(anchor_feats_list, dim=0)
    kshot_feats_list = []
    for class_id in range(n_way):
        anchor_indices_for_class = (anchor_labels_tensor == class_id).nonzero().squeeze(-1)
        if len(anchor_indices_for_class) == 0:
            kshot_feats_list.append(P_name[class_id, :].unsqueeze(0))
        else:
            anchor_feats_for_class = anchor_feats[anchor_indices_for_class]
            prototype_c = F.normalize(torch.mean(anchor_feats_for_class, dim=0, keepdim=True), p=2, dim=-1)
            kshot_feats_list.append(prototype_c)
    P_kshot = torch.cat(kshot_feats_list, dim=0)
    k_min = args.s2_k_min; k_max = args.s2_k_max; current_k = args.k_shot
    alpha_k = 0.0
    if (k_max - k_min) > 0:
        alpha_k = max(0.0, min(1.0, (current_k - k_min) / (k_max - k_min)))
    elif current_k >= k_max:
        alpha_k = 1.0
    if alpha_k == 0.0:
        P_final = P_name
    elif alpha_k == 1.0:
        P_final = P_kshot
    else:
        P_final = (1.0 - alpha_k) * P_name + alpha_k * P_kshot
        P_final = F.normalize(P_final, p=2, dim=-1)
    train_texts = [str(data_obj.raw_text[idx]) for idx in train_indices]
    if not train_texts:
        return None, None, None, None, None, summary
    train_feats_list = []
    for i in range(0, len(train_texts), batch_size):
        batch = train_texts[i:i+batch_size]
        tokens = tokenize(batch, context_length=args.context_length).to(device)
        feats = F.normalize(pretrain_model_frozen.encode_text(tokens), p=2, dim=-1)
        train_feats_list.append(feats)
    train_feats = torch.cat(train_feats_list, dim=0)
    confidences = train_feats @ P_final.t()
    top_indices_list, top_labels_list, top_true_labels_list = [], [], []
    bot_indices_list, bot_true_labels_list = [], []
    all_true_labels = data_obj.y.cpu().numpy()
    for class_id in range(n_way):
        confidence_for_class = confidences[:, class_id]
        if top_k_per_class > 0:
            k_to_use = min(top_k_per_class, len(confidence_for_class))
            top_k_conf, top_k_indices_in_train = torch.topk(confidence_for_class, k=k_to_use)
            threshold_mask = top_k_conf > args.s2_threshold
            final_indices_in_train = top_k_indices_in_train[threshold_mask]
            if len(final_indices_in_train) > 0:
                selected_labels = torch.full((len(final_indices_in_train),), class_id, dtype=torch.long).to(device)
                original_node_indices = train_indices[final_indices_in_train.cpu().numpy()]
                true_labels_original_batch = all_true_labels[original_node_indices]
                top_indices_list.append(original_node_indices)
                top_labels_list.append(selected_labels)
                top_true_labels_list.append(torch.tensor(true_labels_original_batch, dtype=torch.long).to(device))
                true_labels_mapped = [class_mapping.get(l, -1) for l in true_labels_original_batch]
                true_labels_mapped_np = np.array(true_labels_mapped)
                pseudo_labels_np = selected_labels.cpu().numpy()
                id_mask = (true_labels_mapped_np != -1)
                summary['s2_top_noise'] += np.sum(true_labels_mapped_np == -1)
                summary['s2_top_id_total'] += np.sum(id_mask)
                summary['s2_top_total'] += len(final_indices_in_train)
                if np.sum(id_mask) > 0:
                    summary['s2_top_id_correct'] += np.sum(pseudo_labels_np[id_mask] == true_labels_mapped_np[id_mask])
        if bot_k_per_class > 0:
            k_to_use_neg = min(bot_k_per_class, len(confidence_for_class))
            bot_k_conf, bot_k_indices_in_train = torch.topk(confidence_for_class, k=k_to_use_neg, largest=False)
            selected_train_node_ids_neg = train_indices[bot_k_indices_in_train.cpu().numpy()]
            bot_indices_list.append(selected_train_node_ids_neg)
            bot_true_labels_list.append(all_true_labels[selected_train_node_ids_neg])
    if not top_indices_list:
        final_top_indices = final_top_labels = final_top_true_labels = None
    else:
        final_top_indices = np.concatenate(top_indices_list)
        final_top_labels = torch.cat(top_labels_list, dim=0).cpu().numpy()
        final_top_true_labels = torch.cat(top_true_labels_list, dim=0).cpu().numpy()
    if not bot_indices_list:
        final_bot_indices = final_bot_true_labels = None
    else:
        final_bot_indices = np.concatenate(bot_indices_list)
        final_bot_true_labels = np.concatenate(bot_true_labels_list)
        summary['s2_bot_total'] = len(final_bot_indices)
    return final_top_indices, final_top_labels, final_top_true_labels, final_bot_indices, final_bot_true_labels, summary
# =========================
# OOD concept retrieval (merged from retrieval_utils.py)
# =========================
DATASET_DOMAIN_MAP = {
    "cora": "scientific papers in Artificial Intelligence and Machine Learning",
    "citeseer": "scientific papers in Computer and Information Science",
    "wikics": "Computer Science articles from Wikipedia",
    "pubmed": "scientific papers in Medical and Biological Science",
    "arxiv": "scientific papers across various fields like Physics, Mathematics, and Computer Science",
    "ogbn_arxiv": "scientific papers across various fields like Physics, Mathematics, and Computer Science",
    "reddit": "social media discussions, online forums, and community posts",
    "history": "historical literature descriptions, events and biographies",
    "children": "children learning, education and parenting related topics",
    "default": "the provided topics"
}
PROMPT_TEMPLATES = {
    "default_balanced": {
        "strategy": "balanced",
        "prompt_near": (
            "You are an expert in {id_domain_description}.\n"
            "I am working on a task where the 'in-distribution' (ID) topics are: [{id_class_str}].\n"
            "I need a list of {near_ood_count} *other* specific technical concepts, algorithms, and subfields from the general domain of {id_domain_description}.\n"
            "CRITICAL: Your list MUST NOT contain any of the ID topics ({id_class_str}) or their direct sub-topics.\n"
            "Your output MUST be a clean list, with each term on a new line. No numbering or explanations."
        ),
        "prompt_far": (
            "You are a general knowledge expert.\n"
            "The 'in-distribution' (ID) topics for my task are from the domain of '{id_domain_description}'.\n"
            "I need a list of {far_ood_count} 'out-of-distribution' (OOD) candidate terms.\n"
            "These OOD terms MUST be specific technical concepts from fields **NOT related to {id_domain_description}**.\n"
            "Good examples: Operating Systems, Renaissance Art, Macroeconomics, Constitutional Law, Organic Chemistry.\n"
            "Your output MUST be a clean list, with each term on a new line. No numbering or explanations."
        )
    }
}

def _clean_llm_output(text: str) -> List[str]:
    text = re.sub(r'^\s*here.*?is a list of.*?:?\s*\n*', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'```[\w\s]*', '', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    lines = text.strip().split('\n'); cleaned_terms = []
    for line in lines:
        line = line.replace('_', ' ').replace('-', ' ')
        term = re.sub(r'^\s*[-\*\d]+\.\s*', '', line.strip())
        term = re.sub(r'\s*\(.*?\)\s*', '', term).strip()
        if re.match(r'^[a-zA-Z\s]+$', term) and 3 < len(term) < 50:
            cleaned_terms.append(' '.join(term.split()))
    return sorted(list(set(cleaned_terms)))

def get_llm_candidates_hierarchical(id_class_names: List[str], count: int, domain: str, args: object) -> List[str]:
    id_class_str = ", ".join(f"'{name}'" for name in id_class_names)
    id_domain_description = DATASET_DOMAIN_MAP.get(domain, DATASET_DOMAIN_MAP["default"])
    config = PROMPT_TEMPLATES["default_balanced"]
    all_candidates = set()
    near_ood_count = count // 2
    far_ood_count = count - near_ood_count
    if near_ood_count > 0:
        prompt_near_str = config["prompt_near"].format(id_class_str=id_class_str, id_domain_description=id_domain_description, near_ood_count=near_ood_count)
        response_near = call_llm_api([{"role": "user", "content": prompt_near_str}], debug=False)
        if response_near: all_candidates.update(_clean_llm_output(response_near))
    if far_ood_count > 0:
        prompt_far_str = config["prompt_far"].format(id_class_str=id_class_str, id_domain_description=id_domain_description, far_ood_count=far_ood_count)
        response_far = call_llm_api([{"role": "user", "content": prompt_far_str}], debug=False)
        if response_far: all_candidates.update(_clean_llm_output(response_far))
    final_candidates = sorted(list(all_candidates))
    return final_candidates

def get_llm_candidates(id_class_names: List[str], count: int, domain: str, args: object) -> List[str]:
    cache_dir = './llm_cache'; os.makedirs(cache_dir, exist_ok=True)
    filename_key = f"{args.n_way}way"
    cache_filename = f"{domain}_{filename_key}_{count}_default_balanced.json"
    cache_path = os.path.join(cache_dir, cache_filename)
    if args.use_llm_cache and os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
        if cached_data and cached_data != ["LLM API REQUEST ERROR"]:
            print("Successfully loaded LLM candidates from cache.")
            return cached_data
        else:
            print(f"Warning: Found invalid or error-containing cache file at {cache_path}. Regenerating...")

    print("Generating new LLM candidates...")
    llm_candidates = get_llm_candidates_hierarchical(id_class_names, count, domain, args)
    if not llm_candidates:
        print("Warning: LLM candidate generation failed. Writing error to cache.")
        llm_candidates = ["LLM API REQUEST ERROR"]

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(llm_candidates, f, indent=4)

    if llm_candidates == ["LLM API REQUEST ERROR"]:
        return [] # Return empty list to signal failure downstream

    return llm_candidates
    if args.use_llm_cache:
        try:
            candidates = [fn for fn in os.listdir(cache_dir) if fn.startswith(f"{domain}_") and fn.endswith(f"_{count}_default_balanced_v1.json")]
        except FileNotFoundError:
            candidates = []
        if candidates:
            fallback_file = os.path.join(cache_dir, candidates[0])
            with open(fallback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return []
    return llm_candidates

def get_balanced_ood_concept_pool(clip_model: GraphTextEncoder, id_class_names: list, args, device) -> List[str]:
    clip_model.eval()
    llm_candidates = get_llm_candidates(id_class_names, args.llm_candidate_count, args.data_name, args)
    if not llm_candidates:
        return []
    with torch.no_grad():
        id_prompts = [f"a photo of a {name.replace('_', ' ')}" for name in id_class_names]
        id_tokens = tokenize(id_prompts, context_length=args.context_length).to(device)
        id_text_features = F.normalize(clip_model.encode_text(id_tokens), p=2, dim=-1)
        all_candidate_features_list = []
        batch_size = 512
        desc = "Encoding LLM candidates"
        if not (args.use_llm_cache and os.path.exists(os.path.join('./llm_cache', f"{args.data_name}_{args.n_way}way_{args.llm_candidate_count}_default_balanced.json"))):
            desc = "Encoding new LLM candidates"

        for i in tqdm(range(0, len(llm_candidates), batch_size), desc=desc):
            batch_words = llm_candidates[i:i+batch_size]
            prompts = [f"a photo of a {word}" for word in batch_words]
            tokens = tokenize(prompts, context_length=args.context_length).to(device)
            text_features = clip_model.encode_text(tokens)
            all_candidate_features_list.append(text_features)
        all_candidate_features = torch.cat(all_candidate_features_list, dim=0)
        all_candidate_features_norm = F.normalize(all_candidate_features, p=2, dim=-1)
        sim_matrix = all_candidate_features_norm @ id_text_features.t()
        max_sim_to_id, _ = torch.max(sim_matrix, dim=1)
        num_to_return = args.retrieval_p
        if num_to_return == 0:
            return []
        num_near = num_to_return // 2
        num_far = num_to_return - num_near
        _, sorted_indices_near = torch.sort(max_sim_to_id, descending=True)
        near_ood_words = [llm_candidates[i] for i in sorted_indices_near.cpu().numpy()][:num_near]
        _, sorted_indices_far = torch.sort(max_sim_to_id, descending=False)
        far_ood_words = [llm_candidates[i] for i in sorted_indices_far.cpu().numpy()][:num_far]
        final_ood_words = sorted(list(set(near_ood_words + far_ood_words)))
    return final_ood_words

# Hybrid negative mining (exposed for training)
@torch.no_grad()
def get_hybrid_negatives(model, data_obj, unlabeled_indices, id_class_names, ood_words, num_negatives: int, id_protection_threshold: float, ood_similarity_threshold: float, args, device, quiet: bool = True):
    model.eval()
    id_prompts = [f"a photo of a {name.replace('_', ' ')}" for name in id_class_names]
    id_tokens = tokenize(id_prompts, context_length=args.context_length).to(device)
    id_prototypes = F.normalize(model.encode_text(id_tokens), p=2, dim=-1)
    ood_prompts = [f"a photo of a {word.replace('_', ' ')}" for word in ood_words]
    ood_tokens = tokenize(ood_prompts, context_length=args.context_length).to(device)
    ood_prototypes = F.normalize(model.encode_text(ood_tokens), p=2, dim=-1)
    unlabeled_texts = [str(data_obj.raw_text[i]) for i in unlabeled_indices]
    batch_size = 512
    unlabeled_feats_list = []
    for i in range(0, len(unlabeled_texts), batch_size):
        batch = unlabeled_texts[i:i+batch_size]
        tokens = tokenize(batch, context_length=args.context_length).to(device)
        feats = F.normalize(model.encode_text(tokens), p=2, dim=-1)
        unlabeled_feats_list.append(feats)
    unlabeled_feats = torch.cat(unlabeled_feats_list, dim=0)
    sim_to_id = unlabeled_feats @ id_prototypes.t(); max_sim_id, _ = torch.max(sim_to_id, dim=1)
    sim_to_ood = unlabeled_feats @ ood_prototypes.t(); max_sim_ood, _ = torch.max(sim_to_ood, dim=1)
    id_protection_mask = max_sim_id < id_protection_threshold
    ood_similarity_mask = max_sim_ood > ood_similarity_threshold
    pure_negative_mask = id_protection_mask & ood_similarity_mask
    candidate_indices = torch.where(pure_negative_mask)[0]
    if len(candidate_indices) == 0:
        return np.array([], dtype=np.int64)
    if len(candidate_indices) > num_negatives:
        final_indices_in_unlabeled = candidate_indices[torch.randperm(len(candidate_indices))[:num_negatives]]
    else:
        final_indices_in_unlabeled = candidate_indices
    final_negative_indices = unlabeled_indices[final_indices_in_unlabeled.cpu().numpy()]
    return final_negative_indices

# =========================
# Training-stage data preparation (merged from data_loader.py)
# =========================
def prepare_ood_words(pretrain_model_frozen, id_class_names_to_use, args, device):
    return get_balanced_ood_concept_pool(pretrain_model_frozen, id_class_names_to_use, args, device)


def build_loaders_from_augmented_sets(s1_top_indices, s1_top_labels, s1_top_true_labels, s1_bot_indices, s2_top_indices, s2_top_labels, s2_top_true_labels, s2_bot_indices, support_indices_k_shot, support_labels_k_shot, args):
    all_dare_indices_list = []; all_dare_labels_list = []; all_dare_true_labels_list = []
    if s1_top_indices is not None and len(s1_top_indices) > 0:
        all_dare_indices_list.append(s1_top_indices); all_dare_labels_list.append(s1_top_labels); all_dare_true_labels_list.append(s1_top_true_labels)
    if s2_top_indices is not None and len(s2_top_indices) > 0:
        all_dare_indices_list.append(s2_top_indices); all_dare_labels_list.append(s2_top_labels); all_dare_true_labels_list.append(s2_top_true_labels)
    if not all_dare_indices_list: raise RuntimeError("[TTCM-DARE] DARE (Top-K) dataset is empty.")
    all_dare_indices = np.concatenate(all_dare_indices_list)
    all_dare_labels  = np.concatenate(all_dare_labels_list)
    all_dare_true_labels = np.concatenate(all_dare_true_labels_list)
    dare_batch_size = min(args.ft_batch_size, len(all_dare_indices))
    if dare_batch_size == 0: raise RuntimeError("[TTCM-DARE] DARE (Top-K) dataset has zero samples.")
    dare_dataset = torch.utils.data.TensorDataset(torch.tensor(all_dare_indices, dtype=torch.long), torch.tensor(all_dare_labels, dtype=torch.long), torch.tensor(all_dare_true_labels, dtype=torch.long))
    dare_loader = torch.utils.data.DataLoader(dare_dataset, batch_size=dare_batch_size, shuffle=True, drop_last=True)
    all_neg_indices_list = []
    if s1_bot_indices is not None and len(s1_bot_indices) > 0: all_neg_indices_list.append(s1_bot_indices)
    if s2_bot_indices is not None and len(s2_bot_indices) > 0: all_neg_indices_list.append(s2_bot_indices)
    if not all_neg_indices_list: raise RuntimeError("[TTCM-DARE] NEG (Bottom-K) dataset is empty.")
    all_neg_indices = np.concatenate(all_neg_indices_list)
    neg_batch_size = min(args.ft_batch_size, len(all_neg_indices))
    if neg_batch_size == 0: raise RuntimeError("[TTCM-DARE] NEG (Bottom-K) dataset has zero samples.")
    neg_dataset = torch.utils.data.TensorDataset(torch.tensor(all_neg_indices, dtype=torch.long))
    neg_loader = torch.utils.data.DataLoader(neg_dataset, batch_size=neg_batch_size, shuffle=True, drop_last=True)
    clean_batch_size = min(args.ft_batch_size, len(support_indices_k_shot))
    if clean_batch_size == 0: raise RuntimeError("[TTCM-DARE] clean K-shot dataset has zero samples.")
    clean_dataset = torch.utils.data.TensorDataset(torch.tensor(support_indices_k_shot, dtype=torch.long), torch.tensor(support_labels_k_shot, dtype=torch.long))
    clean_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=clean_batch_size, shuffle=True, drop_last=True)
    return dare_loader, neg_loader, clean_loader


def prepare_training_data(pretrain_model_frozen, data_obj, id_class_names_to_use, class_mapping, args, device, support_indices_k_shot, support_labels_k_shot, train_indices_pool):
    if getattr(args, 'disable_ood_concept', False):
        retrieved_ood_words = []
    else:
        retrieved_ood_words = prepare_ood_words(pretrain_model_frozen, id_class_names_to_use, args, device)
    s1_top_indices, s1_top_labels, s1_top_true_labels, s1_bot_indices, _, _ = get_structure_based_augmented_set(
        pretrain_model_frozen, data_obj.x, data_obj.edge_index, support_indices_k_shot, support_labels_k_shot, train_indices_pool, args.n_way, args.k_shot, data_obj, class_mapping, args, device, quiet=True)
    s2_top_indices, s2_top_labels, s2_top_true_labels, s2_bot_indices, _, _ = get_semantic_based_augmented_set(
        pretrain_model_frozen, data_obj, train_indices_pool, support_indices_k_shot, support_labels_k_shot, id_class_names_to_use, class_mapping, args, device, quiet=True)
    dare_loader, neg_loader, clean_loader = build_loaders_from_augmented_sets(
        s1_top_indices, s1_top_labels, s1_top_true_labels, s1_bot_indices, s2_top_indices, s2_top_labels, s2_top_true_labels, s2_bot_indices, support_indices_k_shot, support_labels_k_shot, args)
    return retrieved_ood_words, dare_loader, neg_loader, clean_loader

