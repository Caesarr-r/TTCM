import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import random
import time
from tqdm import tqdm

# Add project root to Python path to allow absolute imports from 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import core TTCM pretraining modules
from src.models.encoders import GraphTextEncoder as CLIP, tokenize
# Import two data loaders (migrated into src.data.data_module)
from src.data.data_module import DataHelper, InBatchDataset, load_data_from_pt

# AMP
from contextlib import nullcontext
from torch.cuda.amp import autocast, GradScaler

def setup_seed(seed_val):
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _pretokenize_all_texts(raw_text_list, context_length: int, chunk: int = 4096):
    """
    Pre-tokenize all raw texts once to avoid repeated CPU/Numpy indexing and tokenization in the loop.
    Returns a LongTensor of shape [N, context_length] on CPU.
    """
    all_tokens = []
    N = len(raw_text_list)
    for i in tqdm(range(0, N, chunk), desc='Pre-tokenizing all texts', leave=False):
        batch_texts = raw_text_list[i:i + chunk]
        toks = tokenize(batch_texts, context_length=context_length)  # LongTensor [B, L] on CPU
        all_tokens.append(toks)
    return torch.cat(all_tokens, dim=0)

def pretrain_main(args):  # main entry for pretraining
    setup_seed(args.seed)
    if hasattr(torch, 'set_float32_matmul_precision'):
        # In PyTorch 2.x this may speed up matmul on some backends
        torch.set_float32_matmul_precision('high')

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f'--- [TTCM] Pre-training on {args.data_name} ({args.gnn_input} dim) ---')
    print(f'Using device: {device}')

    # --- 1. Data loading ---
    # Special handling for some datasets
    name_lower = args.data_name.lower()
    if name_lower in ['arxiv', 'ogbn-arxiv']:
        dataset_pt_path = os.path.join(args.data_dir, 'ogbn_arxiv', f'arxiv_processed.pt')
        try:
            data_obj = torch.load(dataset_pt_path, map_location=device)
            data_obj.x = data_obj.x.to(torch.float32)
            print(f"Successfully loaded data: {dataset_pt_path}")
        except Exception as e:
            print(f"Error loading {dataset_pt_path}: {e}")
            exit(1)
    elif name_lower in ['citeseer', 'history']:
        # Option B: prefer processed features (768-d); otherwise fall back to .npy loading
        processed_path = os.path.join(args.data_dir, name_lower, f'{name_lower}_processed.pt')
        if os.path.exists(processed_path):
            try:
                data_obj = torch.load(processed_path, map_location=device)
                data_obj.x = data_obj.x.to(torch.float32)
                print(f"Successfully loaded processed data: {processed_path}")
            except Exception as e:
                print(f"Error loading processed file ({processed_path}): {e}. Falling back to .npy loader...")
                from data import load_graph_dataset_s2
                data_obj = load_graph_dataset_s2(args.data_name, device, data_root=args.data_dir)
                print(f"Successfully loaded {args.data_name} from .npy files")
        else:
            print(f"Detected dataset {args.data_name}, loading from .npy files...")
            from data import load_graph_dataset_s2
            data_obj = load_graph_dataset_s2(args.data_name, device, data_root=args.data_dir)
            print(f"Successfully loaded {args.data_name} from .npy files")
    else:
        dataset_pt_path = os.path.join(args.data_dir, args.data_name, f'{args.data_name}_processed.pt')
    try:
        data_obj = torch.load(dataset_pt_path, map_location=device)
        data_obj.x = data_obj.x.to(torch.float32)
        print(f"Successfully loaded data: {dataset_pt_path}")
    except FileNotFoundError:
        print(f"Error: Data not found at {dataset_pt_path}.")
        return
    except Exception as e:
        print(f"Error loading {dataset_pt_path}: {e}")
        return

    feature_dim = data_obj.x.shape[1]
    args.gnn_input = feature_dim
    args.gnn_hid = feature_dim
    args.gnn_output = feature_dim
    args.embed_dim = feature_dim
    print(f"Detected feature dimension = {feature_dim}, updated model hyperparameters accordingly.")

    if not hasattr(data_obj, 'raw_text') or data_obj.raw_text is None:
        # Compatibility: use wikics-style clean_text if available
        if hasattr(data_obj, 'clean_text') and data_obj.clean_text is not None:
            data_obj.raw_text = data_obj.clean_text
        else:
            data_obj.raw_text = [f"node {i}" for i in range(data_obj.x.shape[0])]
    else:
        data_obj.raw_text = [str(text) for text in data_obj.raw_text]

    # --- 2. Model initialization ---
    model = CLIP(args, setup_optimizer=True, raw_feature_dim=args.gnn_input).to(device)
    print("Starting Graph-Text Alignment Pre-training...")

    # --- 2.1 AMP and GradScaler configuration ---
    amp_dtype = None
    use_amp = False
    scaler = None
    if device.type == 'cuda' and args.amp_dtype != 'none':
        if args.amp_dtype == 'bf16' and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            use_amp = True
            scaler = None  # bf16 typically does not require GradScaler
        elif args.amp_dtype == 'fp16':
            amp_dtype = torch.float16
            use_amp = True
            scaler = GradScaler()
        else:
            # Fallback: disable AMP
            amp_dtype = None
            use_amp = False
            scaler = None

    # --- 3. Select data loaders based on sampling mode ---
    if args.sampling_mode.startswith('explicit'):
        dataset = DataHelper(data_obj.edge_index.cpu().numpy(), args)
        tokens_all_device = None
    else:  # 'in_batch'
        print("Using [in_batch] mode: InBatchDataset loads nodes only and performs efficient negative sampling on GPU.")
        dataset = InBatchDataset(data_obj.edge_index)

        # Key optimization: pre-tokenize once and keep tokens on GPU to avoid per-step CPU/Numpy overhead
        t0 = time.time()
        tokens_all_cpu = _pretokenize_all_texts(data_obj.raw_text, context_length=args.context_length,
                                               chunk=args.pretok_chunk)
        # Optionally keep on CPU and move by batch if memory is tight; here we favor GPU residency for speed.
        tokens_all_device = tokens_all_cpu.to(device, non_blocking=True)
        del tokens_all_cpu
        torch.cuda.empty_cache()
        t1 = time.time()
        print(f"Pre-tokenization finished and moved to {device}, shape={tuple(tokens_all_device.shape)}, time {t1 - t0:.2f}s")

    # DataLoader configuration
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Force 0 to avoid deadlocks between tqdm and multiprocessing
        drop_last=True,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=False,

    )

    print("\nStarting Contrastive Pre-training...")
    train_start_time = time.time()

    model.train()

    for epoch in range(args.epoch_num):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epoch_num}")
        total_loss_gca = 0
        batch_count = 0

        for sample_batched in pbar:
            if model.optim is None:
                raise RuntimeError("Optimizer is not set.")
            model.optim.zero_grad()

            if args.sampling_mode.startswith('explicit'):
                # Explicit negative sampling logic (covers both explicit and explicit_no_check)
                s_n, neg_n = sample_batched['s_n'], sample_batched['neg_n']
                
                # Robust conversion to numpy arrays and then tensors
                s_n_arr = s_n.numpy() if hasattr(s_n, 'numpy') else np.array(s_n)
                neg_n_arr = neg_n.numpy() if hasattr(neg_n, 'numpy') else np.array(neg_n)
                s_n_on_device = torch.from_numpy(s_n_arr.flatten()).long().to(device, non_blocking=True)

                s_n_text_list = np.array(data_obj.raw_text)[s_n_arr.flatten()].tolist()
                neg_n_text_list = np.array(data_obj.raw_text)[neg_n_arr.flatten()].tolist()

                s_n_text_tokens = tokenize(s_n_text_list, context_length=args.context_length).to(device, non_blocking=True)
                neg_n_text_tokens = tokenize(neg_n_text_list, context_length=args.context_length).to(device, non_blocking=True)

                with autocast(enabled=use_amp, dtype=amp_dtype):
                    s_image_features, s_text_features = model(
                        data_obj.x, data_obj.edge_index, s_n_on_device, s_n_text_tokens
                    )
                    neg_text_features = model.encode_text(neg_n_text_tokens)
                    neg_text_features = neg_text_features.reshape(s_image_features.shape[0], args.neg_num, args.embed_dim)

                    s_image_features = F.normalize(s_image_features, p=2, dim=-1)
                    s_text_features = F.normalize(s_text_features, p=2, dim=-1)
                    neg_text_features = F.normalize(neg_text_features, p=2, dim=-1)

                    logit_scale = model.logit_scale.exp()

                    positive_sim = (s_image_features * s_text_features).sum(dim=-1)
                    negative_sim = (s_image_features.unsqueeze(1) * neg_text_features).sum(dim=-1)

                    positive_exp = torch.exp(positive_sim * logit_scale)
                    negative_exp = torch.exp(negative_sim * logit_scale).sum(dim=1)

                    loss_gca = -torch.log(positive_exp / (positive_exp + negative_exp)).mean()

            elif args.sampling_mode == 'in_batch':
                # Efficient in-batch negative sampling logic
                node_indices = sample_batched.type(torch.long).to(device, non_blocking=True)

                # Key optimization: directly index pre-tokenized tokens that reside on GPU
                node_tokens = tokens_all_device.index_select(0, node_indices)

                if use_amp and scaler is not None:  # fp16 + GradScaler
                    with autocast(dtype=amp_dtype):
                        image_features, text_features = model(data_obj.x, data_obj.edge_index, node_indices, node_tokens)
                        image_features = F.normalize(image_features, p=2, dim=-1)
                        text_features = F.normalize(text_features, p=2, dim=-1)
                        logit_scale = model.logit_scale.exp()
                        logits_per_image = logit_scale * image_features @ text_features.t()
                        logits_per_text = logit_scale * text_features @ image_features.t()
                        labels = torch.arange(len(node_indices), device=device)
                        loss_i = F.cross_entropy(logits_per_image, labels)
                        loss_t = F.cross_entropy(logits_per_text, labels)
                        loss_gca = (loss_i + loss_t) / 2
                else:
                    with autocast(enabled=use_amp, dtype=amp_dtype):
                        image_features, text_features = model(data_obj.x, data_obj.edge_index, node_indices, node_tokens)
                        image_features = F.normalize(image_features, p=2, dim=-1)
                        text_features = F.normalize(text_features, p=2, dim=-1)
                        logit_scale = model.logit_scale.exp()
                        logits_per_image = logit_scale * image_features @ text_features.t()
                        logits_per_text = logit_scale * text_features @ image_features.t()
                        labels = torch.arange(len(node_indices), device=device)
                        loss_i = F.cross_entropy(logits_per_image, labels)
                        loss_t = F.cross_entropy(logits_per_text, labels)
                        loss_gca = (loss_i + loss_t) / 2

            # --- Loss computation ends ---
            # Keep computation graph for non-last batches within an epoch
            is_last_batch = (pbar.n == len(loader) - 1)
            if scaler is not None:  # fp16
                scaler.scale(loss_gca).backward()
                scaler.step(model.optim)
                scaler.update()
            else:
                loss_gca.backward()
                model.optim.step()

            total_loss_gca += loss_gca.item()
            batch_count += 1
            pbar.set_postfix(GCA_Loss=f'{loss_gca.item():.4f}')

        avg_gca = total_loss_gca / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{args.epoch_num} completed, Avg Loss: {avg_gca:.4f}")

    train_end_time = time.time()
    print(f"Pre-training completed. Time elapsed: {train_end_time - train_start_time:.2f} seconds")

    # --- Robust model saving ---
    # Allow custom save root (CLI --save_dir > env PRETRAIN_SAVE_DIR > ./res)
    save_root = getattr(args, 'save_dir', None) or os.environ.get('PRETRAIN_SAVE_DIR', 'res')
    save_dir = os.path.join(save_root, args.data_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'pretrained_model.pkl')

    def robust_save(state):
        import io
        tmp_path = save_path + ".tmp"
        # Helper: atomic write of raw bytes
        def _write_bytes_atomic(bdata: bytes, final_path: str, tmp_p: str):
            # Clean old tmp if present
            try:
                if os.path.exists(tmp_p):
                    os.remove(tmp_p)
            except Exception:
                pass
            with open(tmp_p, 'wb') as f:
                f.write(bdata)
                f.flush()
                os.fsync(f.fileno())
            # Atomic replace
            os.replace(tmp_p, final_path)
        
        # Strategy 1: in-memory zip serialization + atomic write
        try:
            buf = io.BytesIO()
            torch.save(state, buf)  # default zip to memory
            _write_bytes_atomic(buf.getvalue(), save_path, tmp_path)
            print(f"pretrained model saved to: {save_path} (zip, mem)")
            return
        except Exception as e1:
            print(f"[save] zip(in-mem) failed, trying legacy(in-mem): {e1}")
        
        # Strategy 2: in-memory legacy serialization + atomic write
        try:
            buf = io.BytesIO()
            torch.save(state, buf, _use_new_zipfile_serialization=False)
            _write_bytes_atomic(buf.getvalue(), save_path, tmp_path)
            print(f"pretrained model saved to: {save_path} (legacy, mem)")
            return
        except Exception as e2:
            print(f"[save] legacy(in-mem) still failed: {e2}")
            # Final fallback: direct zip-to-file
            try:
                torch.save(state, tmp_path)
                os.replace(tmp_path, save_path)
                print(f"pretrained model saved to: {save_path} (zip, file)")
                return
            except Exception as e3:
                print(f"[save] fallback zip(file) failed: {e3}")
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                raise

    # Ensure saving on CPU to avoid potential CUDA tensor issues during serialization
    cpu_state = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}
    robust_save(cpu_state)


if __name__ == '__main__':
    import sys, traceback
    parser = argparse.ArgumentParser(description='Pre-training')

    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--data_name', type=str, default='cora', help="Dataset name (e.g., cora)")

    # Sampling mode switch
    parser.add_argument('--sampling_mode', type=str, default='in_batch', choices=['explicit', 'in_batch', 'explicit_no_check'],
                        help='`explicit`: slow, CPU-side negative sampling; `in_batch`: fast, GPU in-batch negatives; `explicit_no_check`: fastest but unsafe global random sampling')

    parser.add_argument('--epoch_num', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--neigh_num', type=int, default=3, help='Only used in explicit sampling mode')
    parser.add_argument('--neg_num', type=int, default=5, help='Only used in explicit sampling mode')
    # Optimization: enable parallel data loading via num_workers to speed up training
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--gnn_input', type=int, default=768)
    parser.add_argument('--gnn_hid', type=int, default=768)
    parser.add_argument('--gnn_output', type=int, default=768)
    parser.add_argument('--embed_dim', type=int, default=768)

    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    # Training acceleration and stability
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['none', 'fp16', 'bf16'],
                        help='Automatic mixed precision dtype: bf16 (recommended on A100), fp16 or none')
    parser.add_argument('--pretok_chunk', type=int, default=4096, help='Number of texts per chunk when pre-tokenizing')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='DataLoader prefetch factor (>0 and num_workers>0 to take effect)')
    # Optional: specify root directory for saving weights (default ./res), overridable via PRETRAIN_SAVE_DIR
    parser.add_argument('--save_dir', type=str, default=None)

    args = parser.parse_args()



    try:
        pretrain_main(args)
        print(f"[pretrained.py] finished OK", flush=True)
    except SystemExit as e:
        # Let SystemExit propagate the appropriate exit code
        raise
    except Exception:
        print("[pretrained.py] ERROR:", flush=True)
        traceback.print_exc()
        sys.exit(1)
