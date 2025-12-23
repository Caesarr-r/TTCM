import os
import csv
import json
from datetime import datetime
from typing import Optional


def _safe_tag(tag: str) -> str:
    if tag is None:
        return 'exp'
    return ''.join(c if c.isalnum() or c in ('-', '_') else '-' for c in str(tag))


def _to_jsonable(d: dict) -> dict:
    def ok(v):
        return isinstance(v, (str, int, float, bool)) or v is None
    return {k: v for k, v in d.items() if ok(v)}


class ExperimentLogger:
    """
    Configurable experiment result logger:
    - Append to a global CSV summary (env EXPERIMENT_CSV_FILE, default experiments_TTCM.csv)
    - Append to a global JSONL summary (env EXPERIMENT_JSONL_FILE, default experiments_TTCM.jsonl)
    - Optionally create per-run CSV/JSON files: {exp_tag}_{mode}_seed{seed}_{timestamp}.{csv/json}
      stored under output_dir (env EXPERIMENT_OUTPUT_DIR, default runs/)
    """
    def __init__(self, config, output_dir: Optional[str] = None, per_run_files: Optional[bool] = None):
        self.config = config
        self.output_dir = output_dir or os.environ.get('EXPERIMENT_OUTPUT_DIR', 'runs')
        self.per_run_files = per_run_files if per_run_files is not None else (os.environ.get('SAVE_PER_RUN_FILES', '0') == '1')
        self.summary_csv_path = os.environ.get('EXPERIMENT_CSV_FILE', 'experiments_TTCM.csv')
        self.summary_jsonl_path = os.environ.get('EXPERIMENT_JSONL_FILE', 'experiments_TTCM.jsonl')
        os.makedirs(self.output_dir, exist_ok=True)

    def _build_prefix(self, mode: str) -> str:
        args = self.config
        exp_tag = getattr(args, 'exp_tag', None) or f"{getattr(args, 'data_name', 'dataset')}_{getattr(args, 'k_shot', 'K')}shot"
        seed = getattr(args, 'seed', 'NA')
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        tag = _safe_tag(exp_tag)
        prefix = f"{tag}_{mode}_seed{seed}_{ts}"
        return os.path.join(self.output_dir, prefix)

    def save_metrics(self, metrics: dict, mode: str = 'full') -> dict:
        if hasattr(self.config, '__dict__'):
            cfg_dict = dict(vars(self.config))
        else:
            cfg_dict = {}
        log_row = dict(cfg_dict)
        log_row.update(metrics)
        log_row['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_row.setdefault('dataset', cfg_dict.get('data_name', 'N/A'))
        log_row.setdefault('backbone', 'GNN')
        method_name = 'TTCM (DARE, concurrent)'
        if cfg_dict.get('disable_dare_weights', False):
            method_name = 'TTCM (ablation: w/o DARE weights)'
        log_row.setdefault('method', method_name)

        fieldnames = [
            'timestamp', 'dataset', 'n_way', 'k_shot', 'seed', 'backbone', 'method', 'exp_tag',
            'AUROC', 'AUPR', 'FPR95', 'IND_Accuracy', 'IND_Macro_F1',
            'ft_lr', 'ft_epochs', 'ft_batch_size', 'ft_temp',
            'dare_lr', 'dare_alpha_max', 'dare_m_target_end',
            'lambda_s',
            'tau_quantile', 'clamp_min', 'lambda_clean',
            'lambda_push',
            'lambda_align',
            'disable_dare_weights',
            'pseudo_k_neighbors', 's1_threshold',
            'see_samples_per_class', 's2_threshold',
            's2_k_min', 's2_k_max',
            'pseudo_k_neighbors_neg',
            'see_samples_per_class_neg',
            'retrieval_p', 'llm_candidate_count',
        ]
        if 'stage1_epochs' in log_row:
            log_row.pop('stage1_epochs', None)
        for k in fieldnames:
            if k not in log_row:
                log_row[k] = log_row.get(k, cfg_dict.get(k, 'N/A'))

        run_csv = run_json = None
        if self.per_run_files:
            prefix = self._build_prefix(mode)
            run_csv = f"{prefix}.csv"
            run_json = f"{prefix}.json"
            with open(run_csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerow({k: log_row.get(k, 'N/A') for k in fieldnames})
            json_payload = {
                'config': _to_jsonable(cfg_dict),
                'metrics': _to_jsonable(metrics),
                'meta': {
                    'mode': mode,
                    'timestamp': log_row['timestamp'],
                }
            }
            with open(run_json, 'w', encoding='utf-8') as f:
                json.dump(json_payload, f, ensure_ascii=False, indent=2)

        agg_csv = self.summary_csv_path
        agg_exists = os.path.isfile(agg_csv)
        with open(agg_csv, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not agg_exists:
                w.writeheader()
            w.writerow({k: log_row.get(k, 'N/A') for k in fieldnames})

        jsonl_payload = {
            'config': _to_jsonable(cfg_dict),
            'metrics': _to_jsonable(metrics),
            'meta': {
                'mode': mode,
                'timestamp': log_row['timestamp'],
            }
        }
        with open(self.summary_jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(jsonl_payload, ensure_ascii=False) + '\n')

        return {
            'run_csv': run_csv,
            'run_json': run_json,
            'summary_csv': agg_csv,
            'summary_jsonl': self.summary_jsonl_path,
        }

