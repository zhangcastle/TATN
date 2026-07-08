#!/usr/bin/env python3
"""
Diagnostic tool for TATN pseudo-label quality analysis.

For each (source_temp, target_temp, cycle) combination, runs the pretrained
source model on the target cycle and reports:
  - full prediction error (vs true labels, for debugging only)
  - how many points survive the stability filter
  - how many valid monotone segments are found
  - per-segment lengths and filtered prediction error

Outputs results to a CSV file. True labels are ONLY used for diagnostics;
they must NOT enter transfer training.

Usage:
    python scripts/diagnose_pseudo_labels.py \\
        --source_temp 10 \\
        --target_temps 25 0 n10 n20 \\
        --model_path ./models/pre-10.pt \\
        --data_path ./normalized_data/Pan/ \\
        --output_csv ./pseudo_diagnostics.csv
"""
import sys
import os
import csv
import math
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow running from repo root or scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import conv, lstm, fc, regression, init_seed
from mydata import Mydataset, Pan_data_path, Pan_all_set


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_source_model(model_path, device):
    models = {
        'conv': conv(), 'lstm': lstm(),
        'fc': fc(), 'regression': regression(),
    }
    for k in models:
        models[k].to(device)
        models[k].eval()
    ckpt = torch.load(model_path, map_location=device)
    for m in ['conv', 'lstm', 'fc', 'regression']:
        models[m].load_state_dict(ckpt[m])
    seed = ckpt.get('seed', 0)
    init_seed(seed)
    print(f'  [model] Loaded {model_path}  (seed={seed})')
    return models


def predict_cycle(models_dict, device, data_path, temp, cycle_file):
    """
    Run source model on one target cycle.
    Returns (y_predict, y_true) as flat numpy arrays.
    y_true is only used for diagnostics — never for training.
    """
    test_data = Mydataset(data_path, temp, [cycle_file], mode='test')
    if len(test_data) == 0:
        return np.array([]), np.array([])
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.squeeze().to(device)
            y_pred = models_dict['regression'](
                         models_dict['fc'](
                             models_dict['lstm'](
                                 models_dict['conv'](x_test)))).squeeze()
            preds.append(y_pred.cpu().numpy().flatten())
            trues.append(y_test.squeeze().numpy().flatten())
    return np.concatenate(preds), np.concatenate(trues)


# ---------------------------------------------------------------------------
# Filtering helpers (mirrors pseudo.py logic exactly)
# ---------------------------------------------------------------------------

def stability_filter(y_predict, pseudo_thresh=0.08, pseudo_window=200):
    """
    Returns (idx_list, idx_set) — stable indices sorted in time order.
    Mirrors the filtering in pseudo.main().
    """
    x = pseudo_window
    y_diff = []
    for i in range(len(y_predict)):
        if (i + x) > len(y_predict) - 1:
            break
        a = ((y_predict[i] - y_predict[i + x]) +
             (y_predict[i] - y_predict[i + int(x / 3)]) +
             (y_predict[i] - y_predict[i + int(2 * x / 3)])) / 2
        y_diff.append(abs(a))
    y_diff = np.asarray(y_diff)
    idx_set = set(range(len(y_diff)))
    for i, e in enumerate(y_diff):
        if e > pseudo_thresh:
            b1 = int(x / 10)
            b2 = int(x)
            for j in range(max(0, i - b1), min(len(y_diff), i + b2)):
                idx_set.discard(j)
    return sorted(idx_set), idx_set


def segment_analysis(idx_list, idx_set, y_predict, pseudo_thresh=0.08, min_segment_len=90):
    """
    Returns list of valid segment sizes (mirrors pseudo.main() segment logic).
    """
    if len(idx_list) < 2:
        return []
    y_predict_cut = y_predict[idx_list]
    y_diff2 = -(np.diff(y_predict_cut))
    split_point = [0]
    for i, e in enumerate(y_diff2):
        if abs(e) > pseudo_thresh:
            split_point.append(i)

    segment_sizes = []
    for i in range(len(split_point)):
        if i == len(split_point) - 1:
            beg = min(split_point[i] + 2, len(idx_list) - 2)
            end = len(idx_list) - 2
        else:
            beg = split_point[i] + 2
            end = split_point[i + 1] - 2
            if y_diff2[split_point[i + 1]] < 0:
                continue
        if beg < 0 or end < 0 or beg >= len(idx_list) or end >= len(idx_list) or beg >= end:
            continue
        start_idx = idx_list[beg]
        end_idx   = idx_list[end]
        if end_idx <= start_idx:
            continue
        split_set = {k for k in range(start_idx, end_idx) if k in idx_set}
        if len(split_set) >= min_segment_len:
            segment_sizes.append(len(split_set))
    return segment_sizes


# ---------------------------------------------------------------------------
# Main diagnostic loop
# ---------------------------------------------------------------------------

def diagnose(source_temp, target_temps, model_path, data_path, all_cycles,
             output_csv, pseudo_thresh=0.08, pseudo_window=200, min_segment_len=90):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n=== TATN Pseudo-Label Diagnostics ===')
    print(f'Source: {source_temp}°C    Device: {device}')
    print(f'Filter: thresh={pseudo_thresh}  window={pseudo_window}  min_seg={min_segment_len}\n')

    models_dict = load_source_model(model_path, device)

    rows = []
    for target_temp in target_temps:
        print(f'\n--- Target: {target_temp}°C ---')
        for cycle in all_cycles:
            cycle_name = cycle.replace('.mat', '')
            print(f'  {cycle_name} ...', end=' ', flush=True)

            y_pred, y_true = predict_cycle(models_dict, device, data_path, target_temp, cycle)

            if len(y_pred) == 0:
                print('SKIP (no data)')
                continue

            raw_n = len(y_pred)

            # Full-cycle prediction error (diagnostic only, never used for training)
            full_mae  = float(np.mean(np.abs(y_pred - y_true)))
            full_rmse = float(math.sqrt(np.mean((y_pred - y_true) ** 2)))
            full_max  = float(np.max(np.abs(y_pred - y_true)))

            # Stability filter
            idx_list, idx_set = stability_filter(y_pred, pseudo_thresh, pseudo_window)
            kept_n     = len(idx_list)
            kept_ratio = kept_n / raw_n if raw_n > 0 else 0.0

            # Filtered prediction error (diagnostic only)
            if kept_n > 0:
                filt_pred = y_pred[idx_list]
                filt_true = y_true[idx_list]
                filt_mae  = float(np.mean(np.abs(filt_pred - filt_true)))
                filt_rmse = float(math.sqrt(np.mean((filt_pred - filt_true) ** 2)))
                filt_max  = float(np.max(np.abs(filt_pred - filt_true)))
            else:
                filt_mae = filt_rmse = filt_max = float('nan')

            # Segment analysis
            seg_sizes = segment_analysis(idx_list, idx_set, y_pred, pseudo_thresh, min_segment_len)

            print(f'full_mae={full_mae*100:.2f}%  kept={kept_ratio*100:.0f}%({kept_n}/{raw_n})  segs={len(seg_sizes)} {seg_sizes}')

            rows.append({
                'source_temp':            source_temp,
                'target_temp':            target_temp,
                'cycle_name':             cycle_name,
                'raw_num_points':         raw_n,
                'kept_num_points':        kept_n,
                'kept_ratio':             round(kept_ratio, 4),
                'num_segments':           len(seg_sizes),
                'segment_lengths':        str(seg_sizes),
                'full_pseudo_mae':        round(full_mae, 6),
                'full_pseudo_rmse':       round(full_rmse, 6),
                'max_abs_error_full':     round(full_max, 6),
                'filtered_pseudo_mae':    '' if math.isnan(filt_mae)  else round(filt_mae, 6),
                'filtered_pseudo_rmse':   '' if math.isnan(filt_rmse) else round(filt_rmse, 6),
                'max_abs_error_filtered': '' if math.isnan(filt_max)  else round(filt_max, 6),
            })

    if not rows:
        print('\nNo results to save.')
        return

    out_dir = os.path.dirname(os.path.abspath(output_csv))
    os.makedirs(out_dir, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n✓ Saved diagnostics ({len(rows)} rows) → {output_csv}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnose TATN pseudo-label quality')
    parser.add_argument('--source_temp',  type=str, default='10',
                        help='Source temperature key (default: 10)')
    parser.add_argument('--target_temps', type=str, nargs='+', default=['25', '0', 'n10', 'n20'],
                        help='Target temperature keys (default: 25 0 n10 n20)')
    parser.add_argument('--model_path',   type=str, required=True,
                        help='Path to pretrained source .pt checkpoint')
    parser.add_argument('--data_path',    type=str, default=None,
                        help='Root of normalized data dir (default: Pan_data_path from mydata.py)')
    parser.add_argument('--cycles',       type=str, nargs='+', default=None,
                        help='Cycle filenames to evaluate (default: Pan_all_set)')
    parser.add_argument('--output_csv',   type=str, default='./pseudo_diagnostics.csv')
    parser.add_argument('--pseudo_thresh',    type=float, default=0.08)
    parser.add_argument('--pseudo_window',    type=int,   default=200)
    parser.add_argument('--min_segment_len',  type=int,   default=90)
    args = parser.parse_args()

    diagnose(
        source_temp    = args.source_temp,
        target_temps   = args.target_temps,
        model_path     = args.model_path,
        data_path      = args.data_path or Pan_data_path,
        all_cycles     = args.cycles or Pan_all_set,
        output_csv     = args.output_csv,
        pseudo_thresh  = args.pseudo_thresh,
        pseudo_window  = args.pseudo_window,
        min_segment_len= args.min_segment_len,
    )
