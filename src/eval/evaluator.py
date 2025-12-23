import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import rankdata
from src.models.encoders import tokenize

@torch.no_grad()
def evaluate_model(model,
                   test_indices,
                   all_node_features, all_edge_index,
                   id_class_names,
                   ood_words,
                   class_mapping,
                   args, device,
                   test_labels_original=None):
    """
    Prototype-based evaluation for TTCM:
    - ID classification: maximum similarity to ID text prototypes
    - OOD scoring: max_sim_ood - max_sim_id (or rank-transformed variants)
    """
    model.eval()

    test_indices_tensor = torch.tensor(test_indices, dtype=torch.long).to(device)

    # 1. GNN features
    image_features = model.encode_image(test_indices_tensor, all_node_features, all_edge_index)
    if torch.isnan(image_features).any():
        return None, None
    image_features_norm = F.normalize(image_features, p=2, dim=-1)

    # 2. Prototype prompts (aligned with training)
    use_graph_prompts = getattr(args, 'use_graph_prompts', True)
    id_tmpl = (lambda s: f"a graph node about {s}") if use_graph_prompts else (lambda s: f"a photo of a {s}")
    ood_tmpl = (lambda s: f"a graph node about {s}") if use_graph_prompts else (lambda s: f"a photo of a {s}")

    # 2. ID prototypes
    id_prompts = [id_tmpl(name.replace('_', ' ')) for name in id_class_names]
    id_tokens = tokenize(id_prompts, context_length=args.context_length).to(device)
    id_prototypes = model.encode_text(id_tokens)
    id_prototypes_norm = F.normalize(id_prototypes, p=2, dim=-1)

    # 3. OOD prototypes (if enabled)
    if not getattr(args, 'disable_ood_concept', False):
        ood_prompts = [ood_tmpl(word.replace('_', ' ')) for word in ood_words]
        ood_tokens = tokenize(ood_prompts, context_length=args.context_length).to(device)
        ood_prototypes = model.encode_text(ood_tokens)
        ood_prototypes_norm = F.normalize(ood_prototypes, p=2, dim=-1)
    else:
        ood_prototypes_norm = None

    # 4. ID classification
    sim_id = (image_features_norm @ id_prototypes_norm.t())
    id_logits = sim_id / args.ft_temp
    test_preds_mapped = id_logits.argmax(dim=1).cpu().numpy()
    reverse_class_mapping = {v: k for k, v in class_mapping.items()}
    test_preds_original = np.array([reverse_class_mapping.get(p, -99) for p in test_preds_mapped])

    # 5. OOD detection
    max_sim_id, _ = torch.max(sim_id, dim=1)

    if not getattr(args, 'disable_ood_concept', False):
        if ood_prototypes_norm is not None and ood_prototypes_norm.shape[0] > 0:
            sim_ood = (image_features_norm @ ood_prototypes_norm.t())
            max_sim_ood, _ = torch.max(sim_ood, dim=1)
        else:
            max_sim_ood = torch.zeros_like(max_sim_id)
        score_contrastive = max_sim_ood - max_sim_id
    else:
        score_contrastive = -max_sim_id
    rank_contrastive = rankdata(score_contrastive.cpu().numpy(), method='dense')

    energy_scores = (-torch.logsumexp(id_logits, dim=1)).cpu().numpy()
    rank_energy = rankdata(energy_scores, method='dense')

    use_energy = getattr(args, 'eval_use_energy', False)
    final_ood_scores = rank_energy if use_energy else rank_contrastive

    return test_preds_original, final_ood_scores

