from locale import normalize
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
from utils import *
from mydata import *
from models import *
import numpy as np
import math
import scipy.io as sio
import argparse
import os

# ---------------------------------------------------------------------------
# Segment selection helpers — one per filter_mode
# ---------------------------------------------------------------------------

def _select_segments_original(idx_set, idx_list, y_predict, split_thresh):
    """
    Replicate TL-UESTC/TATN original pseudo-label selection as closely as
    possible.  Key differences from 'strict':
      - idx_list is the raw (unsorted) list(idx_set)  → y_diff2 is computed on
        randomly ordered values, matching the original behaviour
      - no min_segment_len check
      - only minimal crash-protection on beg/end bounds
    """
    if len(idx_list) < 3:
        print("WARNING: idx_list too short (<3), skip segmentation.")
        return []

    y_predict_cut = y_predict[idx_list]          # unsorted, like original
    y_diff2 = -(np.diff(y_predict_cut))

    split_point = [0]
    for i, e in enumerate(y_diff2):
        if abs(e) > split_thresh:
            split_point.append(i)

    sets_list = []
    for i, _ in enumerate(split_point):
        if i == len(split_point) - 1:
            beg = min(split_point[i] + 2, len(idx_list) - 2)
            end = len(idx_list) - 2
        else:
            beg = split_point[i] + 2
            end = split_point[i + 1] - 2
            if y_diff2[split_point[i + 1]] < 0:
                continue

        # Basic crash-protection: skip only if index would raise IndexError
        if beg < 0 or end < 0 or beg >= len(idx_list) or end >= len(idx_list):
            print(f"  WARNING(original): out-of-bounds beg={beg} end={end} "
                  f"len={len(idx_list)}, skipping segment.")
            continue

        split_set = set()
        lo, hi = idx_list[beg], idx_list[end]
        for k in range(lo, hi):          # may be empty if lo >= hi (unsorted)
            if k in idx_set:
                split_set.add(k)

        if split_set:                    # only skip truly empty sets
            sets_list.append(split_set)

    return sets_list


def _select_segments_strict(idx_set, idx_list, y_predict, split_thresh,
                             min_segment_len):
    """
    Fork-specific strict mode:
      - sorted idx_list  (our fix for the unsorted-randomness bug)
      - invalid segment checks (bounds, end_idx <= start_idx)
      - min_segment_len filter
    """
    if len(idx_list) < 3:
        print("WARNING: idx_list too short (<3), skip segmentation.")
        return []

    y_predict_cut = y_predict[idx_list]          # sorted
    y_diff2 = -(np.diff(y_predict_cut))

    split_point = [0]
    for i, e in enumerate(y_diff2):
        if abs(e) > split_thresh:
            split_point.append(i)

    sets_list = []
    for i, _ in enumerate(split_point):
        if i == len(split_point) - 1:
            beg = min(split_point[i] + 2, len(idx_list) - 2)
            end = len(idx_list) - 2
        else:
            beg = split_point[i] + 2
            end = split_point[i + 1] - 2
            if y_diff2[split_point[i + 1]] < 0:
                continue

        if beg < 0 or end < 0 or beg >= len(idx_list) or end >= len(idx_list) or beg >= end:
            print(f"  WARNING(strict): invalid segment beg={beg} end={end} "
                  f"len={len(idx_list)}, skipping.")
            continue

        start_idx = idx_list[beg]
        end_idx   = idx_list[end]

        if end_idx <= start_idx:
            print(f"  WARNING(strict): start_idx={start_idx} >= end_idx={end_idx}, skipping.")
            continue

        split_set = {k for k in range(start_idx, end_idx) if k in idx_set}

        if len(split_set) < min_segment_len:
            print(f"  WARNING(strict): segment too short ({len(split_set)} < {min_segment_len}), skipping.")
            continue

        sets_list.append(split_set)

    return sets_list


# ---------------------------------------------------------------------------
# Main pseudo-label generation function
# ---------------------------------------------------------------------------

def main(temp, model_path, file,
         save_dir='./pseudo_labels', data_path=None,
         filter_mode='strict',
         pseudo_thresh=0.08, pseudo_window=200,
         min_segment_len=90, split_thresh=0.08,
         enable_fallback=False):
    """
    Generate pseudo labels for one unlabelled target cycle.

    filter_mode:
      'original' – mirrors TL-UESTC/TATN: unsorted idx_list, no min_segment_len,
                   minimal crash-protection only.
      'strict'   – our fork: sorted idx_list, bounds + min_segment_len checks.
      'fallback' – same as strict; if no valid segments, saves idx_set-filtered
                   data as a single file (for debugging / ablation only).

    NOTE: target true labels (y_test) are used for diagnostics only.
          Saved pseudo-label files always use y_predict as 'ah'.
    """
    print(f'[pseudo] temp={temp}  file={file}  filter_mode={filter_mode}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdls = {}
    mdls['conv']       = conv()
    mdls['lstm']       = lstm()
    mdls['fc']         = fc()
    mdls['regression'] = regression()
    for k in mdls:
        mdls[k].to(device)
        mdls[k].eval()
    ckpt = torch.load(model_path, map_location=device)
    for m in ['conv', 'lstm', 'fc', 'regression']:
        mdls[m].load_state_dict(ckpt[m])
    seed = ckpt['seed']
    init_seed(seed)
    print(f'[pseudo] loaded model {model_path}  seed={seed}')

    if data_path is None:
        data_path = Pan_data_path

    test_data   = Mydataset(data_path, temp, [file], mode='test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    for _, data in enumerate(test_loader):
        x_test, y_test = data
        x_test = x_test.squeeze().to(device)
        y_test = y_test.squeeze()
        with torch.no_grad():
            y_predict = mdls['regression'](
                            mdls['fc'](
                                mdls['lstm'](
                                    mdls['conv'](x_test)))).squeeze()

    y_predict = y_predict.flatten().cpu().numpy()
    y_test    = y_test.flatten().cpu().numpy()
    raw_n     = len(y_predict)

    # ------------------------------------------------------------------
    # Stage 1: stability filter  (shared by all modes)
    # ------------------------------------------------------------------
    x = pseudo_window
    y_diff = []
    for i in range(raw_n):
        if (i + x) > raw_n - 1:
            break
        a = ((y_predict[i] - y_predict[i + x]) +
             (y_predict[i] - y_predict[i + int(x / 3)]) +
             (y_predict[i] - y_predict[i + int(2 * x / 3)])) / 2
        y_diff.append(abs(a))
    y_diff  = np.asarray(y_diff)
    idx_set = set(range(len(y_diff)))

    for i, e in enumerate(y_diff):
        if e > pseudo_thresh:
            b1 = int(x / 10)
            b2 = int(x)
            for j in range(max(0, i - b1), min(len(y_diff), i + b2)):
                idx_set.discard(j)

    first_stage_kept = len(idx_set)

    # ------------------------------------------------------------------
    # Stage 2: segment selection  (mode-specific)
    # ------------------------------------------------------------------
    if filter_mode == 'original':
        idx_list = list(idx_set)                       # UNSORTED – mirrors original
    else:
        idx_list = sorted(idx_set)                     # sorted – our fix

    if len(idx_list) == 0:
        print(f"[pseudo] WARNING: no indices after stability filter – skipping {file}.")
        _print_stats(file, filter_mode, raw_n, first_stage_kept, [], y_predict, y_test,
                     idx_list, idx_set)
        return []

    if filter_mode == 'original':
        sets_list = _select_segments_original(idx_set, idx_list, y_predict, split_thresh)
    else:
        sets_list = _select_segments_strict(idx_set, idx_list, y_predict, split_thresh,
                                            min_segment_len)

    # ------------------------------------------------------------------
    # Stage 3: save pseudo-label files
    # ------------------------------------------------------------------
    out_subdir = os.path.join(save_dir, temp)
    os.makedirs(out_subdir, exist_ok=True)
    saved_files = []

    orig_path = os.path.join(data_path, temp, temp + file)
    mat = sio.loadmat(orig_path)

    for seg_i, split_set in enumerate(sets_list):
        idx_sorted = sorted(split_set)
        y_predict_split = y_predict[idx_sorted]
        # diagnostics only – not saved
        y_test_split    = y_test[idx_sorted]

        current      = mat['current'][idx_sorted]
        voltage      = mat['voltage'][idx_sorted]
        battery_temp = mat['temp'][idx_sorted]
        ah           = y_predict_split.reshape(-1, 1)   # pseudo-labels from model

        fname     = temp + file.replace('.mat', '') + f'_pseudo_{seg_i+1}.mat'
        save_path = os.path.join(out_subdir, fname)
        sio.savemat(save_path, {'current': current, 'voltage': voltage,
                                'temp': battery_temp, 'ah': ah})
        print(f'  saved: {fname}  n={len(idx_sorted)}')
        saved_files.append(file.replace('.mat', '') + f'_pseudo_{seg_i+1}.mat')

    # Fallback: save stability-filtered data when no segments found
    if not saved_files and filter_mode == 'fallback':
        print(f'[pseudo] fallback: saving stability-filtered idx_set data for {file}')
        idx_sorted = sorted(idx_set)
        current      = mat['current'][idx_sorted]
        voltage      = mat['voltage'][idx_sorted]
        battery_temp = mat['temp'][idx_sorted]
        ah           = y_predict[idx_sorted].reshape(-1, 1)
        fname     = temp + file.replace('.mat', '') + '_pseudo_0.mat'
        save_path = os.path.join(out_subdir, fname)
        sio.savemat(save_path, {'current': current, 'voltage': voltage,
                                'temp': battery_temp, 'ah': ah})
        print(f'  saved (fallback): {fname}  n={len(idx_sorted)}')
        saved_files.append(file.replace('.mat', '') + '_pseudo_0.mat')

    # ------------------------------------------------------------------
    # Diagnostics output
    # ------------------------------------------------------------------
    _print_stats(file, filter_mode, raw_n, first_stage_kept, sets_list,
                 y_predict, y_test, idx_list, idx_set)

    return saved_files


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def _print_stats(file, filter_mode, raw_n, first_stage_kept, sets_list,
                 y_predict, y_test, idx_list, idx_set):
    seg_lens  = [len(s) for s in sets_list]
    final_n   = sum(seg_lens)
    kept_r1   = first_stage_kept / max(raw_n, 1)
    kept_r2   = final_n / max(raw_n, 1)

    full_mae  = float(np.mean(np.abs(y_predict - y_test)))
    full_rmse = float(math.sqrt(np.mean((y_predict - y_test) ** 2)))

    all_idx = sorted(set().union(*sets_list)) if sets_list else []
    if all_idx:
        filt_mae  = float(np.mean(np.abs(y_predict[all_idx] - y_test[all_idx])))
        filt_rmse = float(math.sqrt(np.mean((y_predict[all_idx] - y_test[all_idx]) ** 2)))
    else:
        filt_mae = filt_rmse = float('nan')

    print(f"\n[pseudo stats] {file}  filter_mode={filter_mode}")
    print(f"  raw_num_points       : {raw_n}")
    print(f"  first_stage_kept     : {first_stage_kept} ({kept_r1*100:.1f}%)")
    print(f"  num_segments         : {len(sets_list)}")
    print(f"  segment_lengths      : {seg_lens}")
    print(f"  final_kept_num       : {final_n} ({kept_r2*100:.1f}%)")
    print(f"  full_pseudo_mae      : {full_mae*100:.3f}%")
    print(f"  full_pseudo_rmse     : {full_rmse*100:.3f}%")
    if not math.isnan(filt_mae):
        print(f"  filtered_pseudo_mae  : {filt_mae*100:.3f}%")
        print(f"  filtered_pseudo_rmse : {filt_rmse*100:.3f}%")
    else:
        print(f"  filtered_pseudo_mae  : NaN  (no valid segments)")
        print(f"  filtered_pseudo_rmse : NaN")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TATN pseudo-labels')
    parser.add_argument('--temp',     type=str, default='n20')
    parser.add_argument('--model',    type=str, default='models/pre-n10.pt')
    parser.add_argument('--file',     type=str, default='your_file.mat')
    parser.add_argument('--save_dir', type=str, default='./pseudo_labels')
    parser.add_argument('--data_path', type=str, default=None)

    parser.add_argument('--filter_mode', type=str, default='strict',
                        choices=['original', 'strict', 'fallback'],
                        help=(
                            'original: mirrors TL-UESTC/TATN (unsorted, no min_segment_len). '
                            'strict: our fork (sorted, bounds+min_segment_len checks). '
                            'fallback: strict + save stability-filtered data if no segments found.'))
    parser.add_argument('--pseudo_thresh',   type=float, default=0.08,
                        help='Stability filter threshold (default 0.08)')
    parser.add_argument('--pseudo_window',   type=int,   default=200,
                        help='Lookahead window for stability score (default 200)')
    parser.add_argument('--split_thresh',    type=float, default=0.08,
                        help='Threshold for y_diff2 segment splitting (default 0.08)')
    parser.add_argument('--min_segment_len', type=int,   default=90,
                        help='[strict only] Min samples per segment (default 90)')
    parser.add_argument('--enable_fallback', action='store_true',
                        help='Deprecated; use --filter_mode=fallback instead.')
    args = parser.parse_args()

    main(args.temp, args.model, args.file,
         save_dir=args.save_dir, data_path=args.data_path,
         filter_mode=args.filter_mode,
         pseudo_thresh=args.pseudo_thresh,
         pseudo_window=args.pseudo_window,
         min_segment_len=args.min_segment_len,
         split_thresh=args.split_thresh,
         enable_fallback=args.enable_fallback)
