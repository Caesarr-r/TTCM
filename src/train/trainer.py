import torch
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm

from src.models.encoders import tokenize



def _logit(p: float):
    p = min(max(p, 1e-6), 1-1e-6)
    return math.log(p/(1-p))


def _calibrate_tau_alpha_exact(vals_id: torch.Tensor,
                               vals_ood: torch.Tensor,
                               thr_id: float,
                               thr_ood: float,
                               target_ood_rate: float,
                               current_alpha: float,
                               alpha_max: float,
                               fallback_tau: torch.Tensor,
                               device: torch.device):
    if vals_id is None or vals_ood is None or vals_id.numel() == 0 or vals_ood.numel() == 0:
        return fallback_tau, current_alpha, -1.0

    q = torch.quantile(vals_ood, 1.0 - target_ood_rate).item()
    grid = sorted(set([
        max(0.5, current_alpha/2),
        current_alpha,
        (current_alpha + alpha_max)/2,
        alpha_max,
        min(20.0, alpha_max*1.5)
    ]))

    best_alpha = current_alpha
    best_pid = current_alpha * 0 - 1.0
    best_tau = fallback_tau.item() if isinstance(fallback_tau, torch.Tensor) else float(fallback_tau)

    logit_thr = _logit(thr_ood)
    vals_id = vals_id.to(device)
    for a in grid:
        tau_a = q - (logit_thr / max(a, 1e-6))
        r_id = torch.sigmoid(a * (vals_id - tau_a))
        pid = (r_id > thr_id).float().mean().item()
        if pid > best_pid:
            best_pid = pid
            best_alpha = a
            best_tau = tau_a

    return torch.tensor(best_tau, device=device), float(best_alpha), float(best_pid)


def scores_to_weights(s, alpha, tau):
    return torch.sigmoid(alpha * (s - tau))


def _compute_scalar_feats(gnn_feats_norm, text_feats_norm, id_proto_norm, ood_proto_norm):
    cos_gt = (gnn_feats_norm * text_feats_norm).sum(dim=-1, keepdim=True)
    sim_id_g, _ = (gnn_feats_norm @ id_proto_norm.t()).max(dim=1, keepdim=True)
    sim_id_t, _ = (text_feats_norm @ id_proto_norm.t()).max(dim=1, keepdim=True)
    id_aff = torch.maximum(sim_id_g, sim_id_t)
    if ood_proto_norm is not None and ood_proto_norm.shape[0] > 0:
        sim_ood_g, _ = (gnn_feats_norm @ ood_proto_norm.t()).max(dim=1, keepdim=True)
        sim_ood_t, _ = (text_feats_norm @ ood_proto_norm.t()).max(dim=1, keepdim=True)
        ood_aff = torch.maximum(sim_ood_g, sim_ood_t)
    else:
        ood_aff = torch.zeros_like(id_aff)
    margin = id_aff - ood_aff
    return torch.cat([cos_gt, id_aff, ood_aff, margin], dim=-1)


def train_concurrently(model,
                 dare_engine,
                 optimizer_main, optimizer_dare,
                 data_obj,
                 id_class_names, ood_words,
                 args, device,
                 class_mapping,
                 clean_loader,
                 neg_loader,
                 dare_loader,
                 support_indices_k_shot=None,
                 support_labels_k_shot=None):
    TAU_QUANTILE = args.tau_quantile
    CLAMP_MIN = args.clamp_min
    CLAMP_MAX = getattr(args, 'clamp_max', 1.0)
    R_PUSH_MIN = getattr(args, 'r_push_min', 0.2)
    LAMBDA_CLEAN = args.lambda_clean
    LAMBDA_PUSH = args.lambda_push if not getattr(args, 'disable_ood_concept', False) else 0.0
    LAMBDA_ALIGN = args.lambda_align

    dare_r_mode = getattr(args, 'dare_r_mode', 'margin')
    enable_constraint_calibration = getattr(args, 'enable_constraint_calibration', True)
    use_graph_prompts = getattr(args, 'use_graph_prompts', True)
    fix_ood_prototypes = getattr(args, 'fix_ood_prototypes', True)
    decouple_push = getattr(args, 'decouple_push', False)

    id_tmpl = (lambda s: f"a graph node about {s}") if use_graph_prompts else (lambda s: f"a photo of a {s}")
    ood_tmpl = (lambda s: f"a graph node about {s}") if use_graph_prompts else (lambda s: f"a photo of a {s}")

    id_prompts = [id_tmpl(name.replace('_', ' ')) for name in id_class_names]
    id_tokens = tokenize(id_prompts, context_length=args.context_length).to(device)

    ood_prompts = [ood_tmpl(word.replace('_', ' ')) for word in ood_words]
    ood_tokens = tokenize(ood_prompts, context_length=args.context_length).to(device)

    alpha_0 = 0.5
    alpha_max = args.dare_alpha_max

    ood_prototypes_fixed_norm = None
    if fix_ood_prototypes:
        model.eval()
        with torch.no_grad():
            _ood_proto_once = model.encode_text(ood_tokens)
            ood_prototypes_fixed_norm = F.normalize(_ood_proto_once, p=2, dim=-1)
        model.train()

    all_node_features = data_obj.x
    all_edge_index = data_obj.edge_index
    all_raw_text = data_obj.raw_text

    if clean_loader is None or neg_loader is None or dare_loader is None:
        print("Error: [TTCM] clean_loader, neg_loader or dare_loader is None.")
        return

    num_batches_dare = len(dare_loader)
    num_batches_clean = len(clean_loader)
    num_batches_neg = len(neg_loader)

    if num_batches_dare == 0 or num_batches_clean == 0 or num_batches_neg == 0:
        print("Error: [TTCM] one of the data loaders has zero batches.")
        return

    static_tau = torch.tensor(0.0).to(device)

    pbar = tqdm(range(args.ft_epochs), desc="[CGEM Training]")
    for epoch in pbar:
        is_diagnostic_epoch = (epoch % 5 == 0 or epoch == args.ft_epochs - 1)
        t_ratio = epoch / (args.ft_epochs - 1 + 1e-8)
        current_alpha = alpha_0 + (alpha_max - alpha_0) * (t_ratio ** 2)
        alpha_eff = current_alpha
        with torch.no_grad():
            id_prototypes = model.encode_text(id_tokens)
            id_prototypes_norm = F.normalize(id_prototypes, p=2, dim=-1)
            if fix_ood_prototypes and ood_prototypes_fixed_norm is not None:
                ood_prototypes_norm = ood_prototypes_fixed_norm
            else:
                ood_prototypes = model.encode_text(ood_tokens)
                ood_prototypes_norm = F.normalize(ood_prototypes, p=2, dim=-1)
            support_g_embs = model.encode_image(torch.tensor(support_indices_k_shot, dtype=torch.long).to(device), all_node_features, all_edge_index)
            support_labels_tensor = torch.tensor(support_labels_k_shot, dtype=torch.long).to(device)
            id_graph_protos = []
            for c in range(args.n_way):
                mask = (support_labels_tensor == c)
                if mask.sum() > 0:
                    proto = support_g_embs[mask].mean(dim=0)
                else:
                    proto = torch.zeros(args.embed_dim, device=device)
                id_graph_protos.append(proto)
            id_graph_protos_norm = F.normalize(torch.stack(id_graph_protos), p=2, dim=-1)

        def _r_from_margin(margin_tensor, alpha, tau):
            return torch.sigmoid(alpha * (margin_tensor - tau))

        model.eval()
        dare_engine.train()

        optimizer_dare.zero_grad()

        clean_iter_p = iter(clean_loader)
        loss_s_pos_list = []
        for _ in range(num_batches_clean):
            try:
                clean_indices, clean_labels = next(clean_iter_p)
            except StopIteration:
                clean_iter_p = iter(clean_loader)
                clean_indices, clean_labels = next(clean_iter_p)
            clean_indices_dev = clean_indices.to(device)
            with torch.no_grad():
                clean_gnn_feats = model.encode_image(clean_indices_dev, all_node_features, all_edge_index)
                clean_gnn_feats_norm = F.normalize(clean_gnn_feats, p=2, dim=-1)
                clean_texts = [all_raw_text[i] for i in clean_indices.numpy()]
                clean_tokens = tokenize(clean_texts, context_length=args.context_length).to(device)
                clean_text_feats = model.encode_text(clean_tokens)
                clean_text_feats_norm = F.normalize(clean_text_feats, p=2, dim=-1)
                scalar_feats_pos = _compute_scalar_feats(clean_gnn_feats_norm, clean_text_feats_norm, id_prototypes_norm, ood_prototypes_norm)
            s_pos_raw = dare_engine(clean_gnn_feats_norm, clean_text_feats_norm, id_prototypes_norm, id_graph_protos_norm)
            loss_s_pos_list.append(F.binary_cross_entropy_with_logits(s_pos_raw, torch.ones_like(s_pos_raw)))
        loss_s_pos = torch.mean(torch.stack(loss_s_pos_list))

        neg_iter = iter(neg_loader)
        loss_s_neg_list = []
        for _ in range(num_batches_neg):
            try:
                (neg_indices,) = next(neg_iter)
            except StopIteration:
                neg_iter = iter(neg_loader)
                (neg_indices,) = next(neg_iter)
            neg_indices_dev = neg_indices.to(device)
            with torch.no_grad():
                neg_gnn_feats = model.encode_image(neg_indices_dev, all_node_features, all_edge_index)
                neg_gnn_feats_norm = F.normalize(neg_gnn_feats, p=2, dim=-1)
                neg_texts = [all_raw_text[i] for i in neg_indices.numpy()]
                neg_tokens = tokenize(neg_texts, context_length=args.context_length).to(device)
                neg_text_feats = model.encode_text(neg_tokens)
                neg_text_feats_norm = F.normalize(neg_text_feats, p=2, dim=-1)
                scalar_feats_neg = _compute_scalar_feats(neg_gnn_feats_norm, neg_text_feats_norm, id_prototypes_norm, ood_prototypes_norm)
            s_neg_raw = dare_engine(neg_gnn_feats_norm, neg_text_feats_norm, id_prototypes_norm, id_graph_protos_norm)
            loss_s_neg_list.append(F.binary_cross_entropy_with_logits(s_neg_raw, torch.zeros_like(s_neg_raw)))
        loss_s_neg = torch.mean(torch.stack(loss_s_neg_list))

        loss_s_total = args.lambda_s * (loss_s_pos + loss_s_neg)
        if not torch.isnan(loss_s_total):
            loss_s_total.backward()
            optimizer_dare.step()

        dare_engine.eval()
        with torch.no_grad():
            metric_list = []
            clean_iter_tau = iter(clean_loader)
            for _ in range(num_batches_clean):
                try:
                    clean_indices_tau, _ = next(clean_iter_tau)
                except StopIteration:
                    clean_iter_tau = iter(clean_loader)
                    clean_indices_tau, _ = next(clean_iter_tau)
                clean_indices_dev_tau = clean_indices_tau.to(device)
                clean_gnn_feats_s = model.encode_image(clean_indices_dev_tau, all_node_features, all_edge_index)
                clean_gnn_feats_norm_s = F.normalize(clean_gnn_feats_s, p=2, dim=-1)
                clean_texts_s = [all_raw_text[i] for i in clean_indices_tau.numpy()]
                clean_tokens_s = tokenize(clean_texts_s, context_length=args.context_length).to(device)
                clean_text_feats_s = model.encode_text(clean_tokens_s)
                clean_text_feats_norm_s = F.normalize(clean_text_feats_s, p=2, dim=-1)
                scalar_feats_tau = _compute_scalar_feats(clean_gnn_feats_norm_s, clean_text_feats_norm_s, id_prototypes_norm, ood_prototypes_norm)
                s_raw_clean = dare_engine(clean_gnn_feats_norm_s, clean_text_feats_norm_s, id_prototypes_norm, id_graph_protos_norm)
                metric_list.append(s_raw_clean)
            metric_all = torch.cat(metric_list)
            s_raw_clean_all = metric_all.detach().clone()
            if metric_all.numel() > 0:
                static_tau = torch.quantile(metric_all.detach(), TAU_QUANTILE)
            else:
                static_tau = torch.tensor(0.0).to(device)
            if enable_constraint_calibration:
                val_u_list = []
                lbl_u_list = []
                u_iter_cal = iter(dare_loader)
                max_batches = min(4, num_batches_dare)
                for _ in range(max_batches):
                    try:
                        u_idx, _, u_true = next(u_iter_cal)
                    except StopIteration:
                        break
                    u_idx_dev = u_idx.to(device)
                    g_u = model.encode_image(u_idx_dev, all_node_features, all_edge_index)
                    g_u = F.normalize(g_u, p=2, dim=-1)
                    txts_u = [all_raw_text[i] for i in u_idx.numpy()]
                    tok_u = tokenize(txts_u, context_length=args.context_length).to(device)
                    t_u = model.encode_text(tok_u)
                    t_u = F.normalize(t_u, p=2, dim=-1)
                    v = dare_engine(g_u, t_u, id_prototypes_norm, id_graph_protos_norm).detach()
                    val_u_list.append(v)
                    lbl_u_list.append(u_true.clone())
                if val_u_list:
                    vals_u = torch.cat(val_u_list)
                    lbls_u = torch.cat(lbl_u_list)
                    id_class_indices_original = list(class_mapping.keys())
                    is_true_id_u = torch.tensor([x.item() in id_class_indices_original for x in lbls_u], dtype=torch.bool, device=vals_u.device)
                    is_true_ood_u = ~is_true_id_u
                    vals_id = vals_u[is_true_id_u] if torch.sum(is_true_id_u) > 0 else None
                    vals_ood = vals_u[is_true_ood_u] if torch.sum(is_true_ood_u) > 0 else None
                    if getattr(args, 'calibrate_exact', False):
                        static_tau, alpha_eff, _best_pid = _calibrate_tau_alpha_exact(
                            vals_id, vals_ood,
                            thr_id=getattr(args, 'r_thr_id', 0.5),
                            thr_ood=getattr(args, 'r_thr_ood', 0.1),
                            target_ood_rate=getattr(args, 'ood_target_rate', 0.05),
                            current_alpha=current_alpha,
                            alpha_max=alpha_max,
                            fallback_tau=static_tau,
                            device=vals_u.device
                        )
                    else:
                        if torch.sum(is_true_ood_u) > 0:
                            ood_vals = vals_ood
                            q = torch.quantile(ood_vals, 1.0 - getattr(args, 'ood_target_rate', 0.05)).item()
                            thr_ood_local = getattr(args, 'r_thr_ood', 0.1)
                            logit_thr = math.log(thr_ood_local / (1 - thr_ood_local))
                            tau_needed = q - (current_alpha > 0 and (logit_thr / max(current_alpha, 1e-6)) or 0.0)
                            static_tau = torch.tensor(max(static_tau.item(), tau_needed), device=vals_u.device)

        model.train()
        dare_engine.eval()

        optimizer_main.zero_grad()
        total_loss_dare = 0.0
        total_loss_clean = 0.0
        total_loss_push = 0.0
        total_loss_align = 0.0
        r_on_true_id_list = []
        r_on_true_ood_list = []
        cnt_id_over = 0
        cnt_id_total = 0
        cnt_ood_over = 0
        cnt_ood_total = 0
        thr_id = 0.5
        thr_ood = 0.1
        total_loss_main = 0.0
        id_prototypes = model.encode_text(id_tokens)
        id_prototypes_norm = F.normalize(id_prototypes, p=2, dim=-1)
        if not getattr(args, 'disable_ood_concept', False):
            ood_prototypes = model.encode_text(ood_tokens)
            ood_prototypes_norm = F.normalize(ood_prototypes, p=2, dim=-1)
            prototypes_norm = torch.cat([id_prototypes_norm, ood_prototypes_norm], dim=0)
        else:
            ood_prototypes_norm = None
            prototypes_norm = id_prototypes_norm
        dare_iter = iter(dare_loader)
        for i in range(num_batches_dare):
            try:
                batch_indices, batch_labels, batch_true_labels = next(dare_iter)
            except StopIteration:
                dare_iter = iter(dare_loader)
                batch_indices, batch_labels, batch_true_labels = next(dare_iter)
            batch_indices_dev = batch_indices.to(device)
            batch_labels_dev = batch_labels.to(device)
            gnn_features = model.encode_image(batch_indices_dev, all_node_features, all_edge_index)
            gnn_features_norm = F.normalize(gnn_features, p=2, dim=-1)
            batch_texts = [all_raw_text[i] for i in batch_indices.numpy()]
            batch_tokens = tokenize(batch_texts, context_length=args.context_length).to(device)
            text_features = model.encode_text(batch_tokens)
            text_features_norm = F.normalize(text_features, p=2, dim=-1)
            with torch.no_grad():
                g_embs_det = gnn_features_norm.detach()
                t_embs_det = text_features_norm.detach()
                s_raw = dare_engine(g_embs_det, t_embs_det, id_prototypes_norm, id_graph_protos_norm)
                tau = static_tau
                if torch.isnan(s_raw).any():
                    r = torch.full_like(s_raw, 0.5)
                else:
                    r = scores_to_weights(s_raw, alpha_eff, tau)
            if args.disable_dare_weights:
                r = torch.ones_like(s_raw)
            else:
                r = torch.clamp(r, min=CLAMP_MIN, max=CLAMP_MAX)
            r_detached = r.detach()
            logits_g = (gnn_features_norm @ prototypes_norm.t()) / args.ft_temp
            logits_t = (text_features_norm @ prototypes_norm.t()) / args.ft_temp
            loss_g_ce = F.cross_entropy(logits_g, batch_labels_dev, reduction='none', label_smoothing=args.label_smoothing)
            loss_t_ce = F.cross_entropy(logits_t, batch_labels_dev, reduction='none', label_smoothing=args.label_smoothing)
            n_way = id_prototypes_norm.shape[0]
            loss_g_push_ood = (logits_g[:, n_way:]**2).mean(dim=1)
            loss_t_push_ood = (logits_t[:, n_way:]**2).mean(dim=1)
            w_r = r_detached
            w_push = torch.clamp(1.0 - r_detached, min=R_PUSH_MIN)
            loss_dare_g_weighted = (w_r * loss_g_ce + LAMBDA_PUSH * (w_push * loss_g_push_ood)).mean()
            loss_dare_t_weighted = (w_r * loss_t_ce + LAMBDA_PUSH * (w_push * loss_t_push_ood)).mean()
            loss_dare_batch = loss_dare_g_weighted + loss_dare_t_weighted
            if not torch.isnan(loss_dare_batch):
                total_loss_dare += loss_dare_batch.item()
                total_loss_main = total_loss_main + loss_dare_batch
            if LAMBDA_PUSH > 0:
                total_loss_push += (loss_g_push_ood.mean() + loss_t_push_ood.mean()).item()
            if LAMBDA_ALIGN > 0 and not args.disable_dare_weights:
                cos_sim = (gnn_features_norm * text_features_norm).sum(dim=-1)
                loss_align_per_sample = - cos_sim
                loss_align_batch = (r_detached * loss_align_per_sample).mean()
                if not torch.isnan(loss_align_batch):
                    total_loss_align += loss_align_batch.item()
                    total_loss_main = total_loss_main + (LAMBDA_ALIGN * loss_align_batch)
            if is_diagnostic_epoch and r.shape[0] > 0:
                with torch.no_grad():
                    id_class_indices_original = list(class_mapping.keys())
                    is_true_id = torch.tensor([label.item() in id_class_indices_original for label in batch_true_labels], dtype=torch.bool).to(device)
                    true_id_samples = is_true_id
                    true_ood_samples = ~is_true_id
                    r_cur = r.detach()
                    if torch.sum(true_id_samples) > 0:
                        r_on_true_id_list.append(r_cur[true_id_samples].mean().item())
                        cnt_id_total += int(torch.sum(true_id_samples).item())
                        cnt_id_over += int(torch.sum(r_cur[true_id_samples] > thr_id).item())
                    if torch.sum(true_ood_samples) > 0:
                        r_on_true_ood_list.append(r_cur[true_ood_samples].mean().item())
                        cnt_ood_total += int(torch.sum(true_ood_samples).item())
                        cnt_ood_over += int(torch.sum(r_cur[true_ood_samples] > thr_ood).item())
        clean_iter_b = iter(clean_loader)
        for i in range(num_batches_clean):
            try:
                batch_indices, batch_labels = next(clean_iter_b)
            except StopIteration:
                clean_iter_b = iter(clean_loader)
                batch_indices, batch_labels = next(clean_iter_b)
            batch_indices_dev = batch_indices.to(device)
            batch_labels_dev = batch_labels.to(device)
            gnn_features = model.encode_image(batch_indices_dev, all_node_features, all_edge_index)
            gnn_features_norm = F.normalize(gnn_features, p=2, dim=-1)
            batch_texts = [all_raw_text[i] for i in batch_indices.numpy()]
            batch_tokens = tokenize(batch_texts, context_length=args.context_length).to(device)
            text_features = model.encode_text(batch_tokens)
            text_features_norm = F.normalize(text_features, p=2, dim=-1)
            logits_g = (gnn_features_norm @ prototypes_norm.t()) / args.ft_temp
            loss_g_ce_clean = F.cross_entropy(logits_g, batch_labels_dev, reduction='mean', label_smoothing=args.label_smoothing)
            logits_t = (text_features_norm @ prototypes_norm.t()) / args.ft_temp
            loss_t_ce_clean = F.cross_entropy(logits_t, batch_labels_dev, reduction='mean', label_smoothing=args.label_smoothing)
            n_way = id_prototypes_norm.shape[0]
            loss_g_push_clean = (logits_g[:, n_way:]**2).mean()
            loss_t_push_clean = (logits_t[:, n_way:]**2).mean()
            loss_g_clean = loss_g_ce_clean + (LAMBDA_PUSH * loss_g_push_clean)
            loss_t_clean = loss_t_ce_clean + (LAMBDA_PUSH * loss_t_push_clean)
            loss_clean_batch = LAMBDA_CLEAN * (loss_g_clean + loss_t_clean)
            if not torch.isnan(loss_clean_batch):
                total_loss_clean += loss_clean_batch.item()
                total_loss_main = total_loss_main + loss_clean_batch
            if is_diagnostic_epoch and batch_labels_dev.shape[0] > 0:
                with torch.no_grad():
                    id_class_indices_original = list(class_mapping.keys())
                    is_true_id = torch.tensor([label.item() in id_class_indices_original for label in batch_labels_dev], dtype=torch.bool).to(device)
                    true_ood_samples = ~is_true_id
                    g_clean_det = gnn_features_norm.detach()
                    t_clean_det = text_features_norm.detach()
                    s_clean_raw = dare_engine(g_clean_det, t_clean_det, id_prototypes_norm, id_graph_protos_norm).detach()
                    r_clean = scores_to_weights(s_clean_raw, alpha_eff, static_tau)
                    r_clean = torch.clamp(r_clean, min=CLAMP_MIN, max=CLAMP_MAX)
                    if torch.sum(true_ood_samples) > 0:
                        r_on_true_ood_list.append(r_clean[true_ood_samples].mean().item())
                        cnt_ood_total += int(torch.sum(true_ood_samples).item())
                        cnt_ood_over += int(torch.sum(r_clean[true_ood_samples] > thr_ood).item())
        if isinstance(total_loss_main, torch.Tensor) and total_loss_main != 0.0:
            total_loss_main.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_main.step()
        avg_loss_dare = total_loss_dare / num_batches_dare if num_batches_dare > 0 else 0
        avg_loss_clean = total_loss_clean / num_batches_clean if num_batches_clean > 0 else 0
        avg_loss_push = total_loss_push / num_batches_dare if num_batches_dare > 0 else 0
        avg_loss_align = total_loss_align / num_batches_dare if num_batches_dare > 0 else 0
        pbar.set_postfix(
            L_dare=f'{avg_loss_dare:.4f}',
            L_clean=f'{avg_loss_clean:.4f}',
            L_push=f'{avg_loss_push:.4f}',
            L_align=f'{avg_loss_align:.4f}',
            L_s=f'{loss_s_total.item():.4f}',
            tau=f'{static_tau.item():.3f}'
        )
    return

