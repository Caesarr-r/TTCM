import os
import sys
import subprocess
import torch
import argparse

# Add project root to Python path to allow absolute imports from 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can do absolute imports
from src.data.data_module import load_data_from_pt

def run_command(command, dataset_name, step_name):
    """Wrapper for running shell commands with logging and basic error handling."""
    command_str = " ".join(command)
    print(f"\n--- [{dataset_name.upper()}] Start step: {step_name} ---")
    print(f"Command: {command_str}")
    try:
        # REMOVED capture_output=True to allow real-time streaming of logs
        subprocess.run(command, check=True, text=True)
        print(f"--- [{dataset_name.upper()}] Finished step successfully: {step_name} ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"!!! [{dataset_name.upper()}] Step {step_name} failed with return code: {e.returncode} !!!")
        # With streaming output, stdout/stderr are already printed
        return False
    except KeyboardInterrupt:
        print("\n!!! Interrupted by user, exiting. !!!")
        exit(1)

def main(args):
    """Run the unified preparation pipeline for the specified dataset."""
    dataset = args.data_name
    gpu_id = args.gpu
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # --- MOVED LOGIC HERE ---
    # Dynamically set default sampling_mode if not specified by the user
    is_sampling_mode_set_by_user = any(arg.startswith('--sampling_mode') for arg in sys.argv)
    if not is_sampling_mode_set_by_user:
        if dataset.lower() in ['cora', 'citeseer']:
            args.sampling_mode = 'explicit'
        else:
            args.sampling_mode = 'in_batch'

    print(f"--- Start unified preparation for dataset [{dataset.upper()}] on GPU:{gpu_id} ---")

    # Define file paths
    data_dir = os.path.join('dataset', dataset)
    raw_file_path = os.path.join(data_dir, f'{dataset}.pt')
    processed_file_path = os.path.join(data_dir, f'{dataset}_processed.pt')
    pretrained_model_path = os.path.join('res', dataset, 'pretrained_model.pkl')

    # --- Step 1: SBERT feature enhancement ---
    if os.path.exists(processed_file_path):
        print(f"Detected existing processed file {processed_file_path}, skipping SBERT enhancement.")
    else:
        print(f"--- [{dataset.upper()}] Step: SBERT feature enhancement ---")
        try:
            print(f"Loading raw data from {raw_file_path}...")
            data = torch.load(raw_file_path)
            
            from sentence_transformers import SentenceTransformer
            model_name = 'all-mpnet-base-v2'  # 768-dim
            sbert_model = SentenceTransformer(model_name, device=device)
            
            raw_texts = [str(text) for text in data.raw_text]
            new_features = sbert_model.encode(raw_texts, show_progress_bar=True, convert_to_tensor=True, device=device)
            
            data.x = new_features.float().cpu()
            os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
            torch.save(data, processed_file_path)
            print(f"Successfully generated and saved processed dataset to: {processed_file_path}")

        except Exception as e:
            print(f"!!! [{dataset.upper()}] SBERT enhancement step failed: {e} !!!")
            import traceback
            traceback.print_exc()
            return

    # --- Step 2: model pretraining ---
    if os.path.exists(pretrained_model_path):
        print(f"Detected existing pretrained model {pretrained_model_path}, skipping pretraining step.")
    else:
        pretrain_command = [
            'python',
            '-u',
            os.path.join(os.path.dirname(__file__), 'pretrain_contrastive.py'),
            '--data_name', dataset,
            '--gpu', str(gpu_id),
            '--batch_size', str(args.batch_size),
            '--sampling_mode', args.sampling_mode,  # This now carries the correct mode
            '--epoch_num', str(args.epoch_num)
        ]
        ok = run_command(pretrain_command, dataset, "model pretraining")
        if not ok:
            return

    print(f"\n{'='*20} Preparation pipeline for dataset [{dataset.upper()}] completed {'='*20}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SBERT enhancement and pretraining for a specified dataset.')
    parser.add_argument('--data_name', type=str, required=True, help='Dataset name (e.g., cora, citeseer, pubmed)')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID to use (e.g., 0, 1, 2)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for pretraining')
    parser.add_argument('--sampling_mode', type=str, default=None, help='(Optional) Override sampling mode: explicit, in_batch')
    parser.add_argument('--epoch_num', type=int, default=8, help='Number of pretraining epochs')
    
    args = parser.parse_args()
    main(args)
