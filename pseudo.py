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

#use pre-trained source model to generate pseudo labels for unlabelled target data
def main(temp, model_path, file, save_dir='./pseudo_labels', data_path=None,
         enable_fallback=False):
    """
    Generate pseudo labels for one unlabelled target cycle.

    Args:
        temp:            target temperature key (e.g. '25', 'n10')
        model_path:      path to pretrained source .pt file
        file:            cycle filename (e.g. 'Cycle_1.mat')
        save_dir:        root dir for saving pseudo-labelled files
                         (saved to save_dir/{temp}/{temp}<name>_pseudo_N.mat)
        data_path:       root of normalised data; defaults to Pan_data_path
        enable_fallback: if True, save idx_set-filtered data when no monotone
                         segments are found. Default False for strict reproduction
                         (disabled fallback preserves TATN pseudo-label selection
                         strategy; enable only for debugging / ablation studies).

    Returns:
        list of saved filenames (relative, for Mydataset file list);
        empty list if no valid segments found and enable_fallback=False.
    """
    print('temp:',temp)
    # Bug fix 1: removed `device = torch.device('cpu')` that was overriding GPU detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    models['conv'] = conv()
    models['lstm'] = lstm()
    models['fc'] = fc()
    models['regression'] = regression()
    # Bug fix 2: renamed loop variable from `model` to `k` to avoid overwriting parameter `model_path`
    for k in models:
        models[k].to(device)
        models[k].eval()
    # Bug fix 3: now correctly uses the parameter, not the loop variable
    ckpt = torch.load(model_path, map_location=device)
    for m in ['conv','lstm','fc','regression']:
        models[m].load_state_dict(ckpt[m])
    seed = ckpt['seed']
    print('load model')
    print('load seed={}'.format(seed))
    init_seed(seed)

    criterion_mae = nn.L1Loss()
    criterion_mse = nn.MSELoss()
    
    if data_path is None:
        data_path = Pan_data_path

    test_set = []
    test_set.append(file)
    print(test_set)
    test_data = Mydataset(data_path, temp, test_set,mode='test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    print('load data')
    for i, data in enumerate(test_loader):
        print('data ',i)
        x_test, y_test = data
        x_test, y_test = x_test.squeeze(), y_test.squeeze()
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        with torch.no_grad():
            y_predict = models['regression']\
                                (models['fc']\
                                (models['lstm']\
                                    (models['conv'](x_test)))).squeeze()
            print(y_predict.shape)
    y_predict = y_predict.flatten().cpu().numpy()
    y_test = y_test.flatten().cpu().numpy()

    
    y_diff = []
    
    x = 200
    for i in range(len(y_predict)):
        if (i+x) > len(y_predict)-1:
            break
        a = ( (y_predict[i] - y_predict[i+x]) + \
            (y_predict[i] - y_predict[i+int(x/3)]) + \
            (y_predict[i] - y_predict[i+int(2*x/3)])
            ) / 2
        y_diff.append(abs(a))
    y_diff = np.asarray(y_diff)
    idx_set = set(list(range(len(y_diff))))
    
    thresh = 0.08

    for i,e in enumerate(y_diff):
        if e > thresh:
            b1 = int(x/10)
            b2 = int(x)
            front_idx = max(0,i-b1)
            back_idx = min(len(y_diff),i+b2)
            for j in range(front_idx,back_idx):
                idx_set.discard(j)

    idx_list = sorted(idx_set)

    if len(idx_list) == 0:
        print(f"WARNING: skip {file}: no indices left after pseudo-label filtering.")
        return []

    y_predict_cut = y_predict[idx_list]
    y_test_cut = y_test[idx_list]
    y_diff_cut = y_diff[idx_list]
    y_diff2 = -(np.diff(y_predict_cut))
    pairs = []
    sets_list = []

    split_point = []
    split_point.append(0)

    for i, e in enumerate(y_diff2):
        if abs(y_diff2[i]) > 0.08:
            split_point.append(i)
            print(i, idx_list[i], y_diff2[i])

    for i, e in enumerate(split_point):

        if i == (len(split_point)-1):
            beg, end = min(split_point[i]+2, len(idx_list)-2), len(idx_list)-2
        else:
            beg, end = split_point[i]+2, split_point[i+1]-2
            if y_diff2[split_point[i+1]] < 0:
                continue

        if beg < 0 or end < 0 or beg >= len(idx_list) or end >= len(idx_list) or beg >= end:
            print(
                f"WARNING: skip invalid pseudo segment in {file}: "
                f"beg={beg}, end={end}, len(idx_list)={len(idx_list)}"
            )
            continue

        start_idx = idx_list[beg]
        end_idx = idx_list[end]

        if end_idx <= start_idx:
            print(
                f"WARNING: skip invalid pseudo segment in {file}: "
                f"start_idx={start_idx}, end_idx={end_idx}"
            )
            continue

        split_set = set()
        print(start_idx, end_idx)

        for k in range(start_idx, end_idx):
            if k in idx_set:
                split_set.add(k)

        min_segment_len = 90
        if len(split_set) < min_segment_len:
            print(
                f"WARNING: skip short pseudo segment in {file}: "
                f"only {len(split_set)} samples < {min_segment_len}"
            )
            continue

        pairs.append((beg, end))
        sets_list.append(split_set)
    
    # fixed: removed '*' from figure names (illegal filename character)
    fig_cut = 'fig_' + temp + file.replace('.mat','') + '_pseudo_cut.jpg'
    fig     = 'fig_' + temp + file.replace('.mat','') + '_pseudo.jpg'
    out_subdir = os.path.join(save_dir, temp)
    os.makedirs(out_subdir, exist_ok=True)
    saved_files = []

    for i,split_set in enumerate(sets_list):
        fig_split = 'fig_' + temp + file.replace('.mat','') + '_pseudo_split_' + str(i+1) + '.jpg'
        plt.figure()
        y_predict_split = y_predict[sorted(split_set)]
        y_test_split = y_test[sorted(split_set)]
        split_diff = abs(y_predict_split - y_test_split)
        plt.plot(y_predict_split,label='predict',color='red')
        plt.plot(y_test_split,label='label',color='black')
        #plt.plot(y_diff,label='diff',color='blue')
        plt.plot(split_diff,label='diff',color='green')
        plt.legend()
        plt.savefig(fig_split)
        plt.close()
        # Bug fix 4: './your fold' was a placeholder; file name had illegal '*'.
        # Now reads from normalised data dir (data_path param), saves to save_dir param.
        orig_path = data_path + temp + '/' + temp + file
        mat = sio.loadmat(orig_path)
        current = mat['current'][sorted(split_set)]
        voltage = mat['voltage'][sorted(split_set)]
        battery_temp = mat['temp'][sorted(split_set)]
        ah = y_predict_split.reshape(-1, 1)
        fname = temp + file.replace('.mat', '') + '_pseudo_' + str(i+1) + '.mat'
        save_path = os.path.join(out_subdir, fname)
        sio.savemat(save_path, {'current': current, 'voltage': voltage, 'temp': battery_temp, 'ah': ah})
        print('saved:', save_path, y_predict_split.shape)
        saved_files.append(file.replace('.mat', '') + '_pseudo_' + str(i+1) + '.mat')

    # If no valid monotone segments found:
    #   enable_fallback=False (default, strict reproduction): skip this cycle, return [].
    #   enable_fallback=True  (debug/ablation only):          save idx_set-filtered data.
    # NOTE: fallback is DISABLED for formal Table IV/V reproduction to preserve
    # TATN's pseudo-label selection strategy. Enabling it relaxes the selection
    # criteria and should be reported separately.
    if not saved_files:
        if enable_fallback:
            print('no segments found, falling back to stability-filtered idx_set (enable_fallback=True)')
            orig_path = data_path + temp + '/' + temp + file
            mat = sio.loadmat(orig_path)
            idx_sorted = sorted(idx_set)
            current = mat['current'][idx_sorted]
            voltage = mat['voltage'][idx_sorted]
            battery_temp = mat['temp'][idx_sorted]
            ah = y_predict[idx_sorted].reshape(-1, 1)
            fname = temp + file.replace('.mat', '') + '_pseudo_0.mat'
            save_path = os.path.join(out_subdir, fname)
            sio.savemat(save_path, {'current': current, 'voltage': voltage, 'temp': battery_temp, 'ah': ah})
            print('saved (fallback):', save_path)
            saved_files.append(file.replace('.mat', '') + '_pseudo_0.mat')
        else:
            print(f'WARNING: no valid segments for {file} (sets_list empty). Skipping pseudo-label. '
                  f'Pass enable_fallback=True to force-save idx_set data for debugging.')

    #draw figures
    

    print('<{}:{}'.format(str(thresh),np.sum(y_diff<thresh)))
    plt.figure()
    plt.plot(y_predict_cut,label='predict',color='red')
    plt.plot(y_test_cut,label='label',color='black')
    #plt.plot(y_diff,label='diff',color='blue')
    plt.plot(y_diff2,label='diff2',color='green')
    plt.legend()
    plt.savefig(fig_cut)
    plt.close()

    plt.figure()
    plt.plot(y_predict,label='predict',color='red')
    plt.plot(y_test,label='label',color='black')
    plt.plot(y_diff,label='diff',color='blue')
    plt.legend()
    plt.savefig(fig)
    plt.close()

    print(y_predict.shape,y_test.shape)
    
    #calculate error
    loss_mse = np.sum((y_predict-y_test)**2) / len(y_test)
    loss_mae = np.sum( abs(y_predict-y_test) ) / len(y_test)
    loss_rmse = math.sqrt(loss_mse)
    loss_max = max(abs(y_predict-y_test))
    error = 'mae = ' + str(loss_mae) + ' rmse = ' + str(loss_rmse) +  ' max = ' + str(loss_max)
    print(data_path,test_set[0])
    print(error)

    loss_mse = np.sum((y_predict_cut - y_test_cut)**2) / len(idx_set)
    loss_mae = np.sum( abs(y_predict_cut - y_test_cut) ) / len(idx_set)
    loss_rmse = math.sqrt(loss_mse)
    loss_max = max(abs(y_predict_cut - y_test_cut))
    error = 'mae = ' + str(loss_mae) + ' rmse = ' + str(loss_rmse) +  ' max = ' + str(loss_max)
    print(data_path,test_set[0])
    print(error)

    return saved_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate pseudo label')
    parser.add_argument('--temp',type=str,default='n20')
    parser.add_argument('--model',type=str,default='models/pre-n10.pt')
    parser.add_argument('--file',type=str,default='your file')
    args = parser.parse_args()
    main(args.temp, args.model, args.file)