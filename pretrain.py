from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
import argparse
from tqdm import tqdm
import math
import ast
import os
from utils import *
from mydata import *
from models import *
eval_interval = 100
batch_size = 6

def pretrain(rundir,source_temp,target_temp,source_data_path,source_train_set,source_test_set,models, criterion, optimizers, batch_size, epochs,eval_interval, seed=0, device_type=('cuda:0' if torch.cuda.is_available() else 'cpu'), ifsave=True, load_model=False, load_model_path='./saved_model/best.pt'):
  loss_min = 10000
  rundir = mkdir(rundir)
  device = torch.device(device_type)
  if load_model:
      load_saved_model(device,models,optimizers,loss_min,seed)
  if torch.cuda.is_available():
    for  model in models:
      models[model].to(device)
  
  init_seed(seed)
  criterion_mae = nn.L1Loss()
  criterion_mse = nn.MSELoss()
  for temp_idx in range(1):
    source_data = Mydataset(source_data_path, source_temp, source_train_set,mode='train')
    source_loader = DataLoader(source_data, batch_size=batch_size, shuffle=True)
    source_test_data = Mydataset(source_data_path, source_temp, source_test_set,mode='test')
    source_test_loader = DataLoader(source_test_data, batch_size=1, shuffle=False)

    loss_iter_domain = []
    loss_iter_predictor = []
    loss_iter_test = []
    loss_iter_mae = []
    loss_iter_rmse = []
    loss_iter_max = []
    loss_iter_domain_acc = []
    loss_train_predictor = 0
    test_len = len(source_test_loader)
    min_max,min_mae,min_rmse = [],[],[]
    for i in range(test_len):
      min_mae.append(1)
      min_rmse.append(1)
      min_max.append(1)
    #checkpoint = torch.load(load_model_path, map_location=device)
     #models['domain_classifier'].load_state_dict(checkpoint['domain_classifier'])
    for epoch in range(epochs):
      ##########
      #train
      ##########
      for model in models:
        models[model].train()
      loss_train = 0
      loss_test = 0
      source_sample = 0
      #tqdm_mix = tqdm(source_loader,desc='epoch '+str(epoch))
      for i, (source_data, source_label) in enumerate(source_loader):
        source_data = source_data.to(device)
        source_label = source_label.to(device)

        for op in optimizers:
          optimizers[op].zero_grad()

        source_features = models['conv'](source_data)
        predict_label = models['regression']\
                          (models['fc']\
                            (models['lstm'](source_features))).squeeze()
        predict_loss = criterion(predict_label,source_label)
        loss = predict_loss
        loss.backward()
        for op in optimizers:
          optimizers[op].step()
        loss_train += loss.item()
        source_sample += len(source_data)
        #if ((epoch+1) % eval_interval) == 0:
        #  plot_result(source_label, predict_label, save_image='train', 
        #          test_name=source_data_path[7:-1] + source_temp + '_epoch_' + str(epoch))
      loss_train = loss_train/(source_sample)
      print('epoch {}:loss {}'.format(epoch, loss_train))
      if (loss_train < loss_min) & (ifsave==True):
        loss_min = loss_train
        path = rundir+'/saved_model/best.pt'
        save_model(models, optimizers, loss_min, seed,path)
        print('min loss:{} saved model'.format(loss_min))
      ##########
      #test
      ##########
      for model in models:
        models[model].eval()
      
      #tqdm_test = tqdm(source_test_loader, desc='source data test')
      loss_mae = 0
      loss_rmse = 0
      loss_max = 0
      if ((epoch+1) % eval_interval) == 0:
            print('source test res')
      for i, data in enumerate(source_test_loader):
        x_test, y_test = data
        x_test, y_test = x_test.squeeze(), y_test.squeeze()
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        with torch.no_grad():
          y_predict = models['regression']\
                            (models['fc']\
                              (models['lstm']\
                                (models['conv'](x_test)))).squeeze()

        loss_mse = criterion_mse(y_predict, y_test)
        loss_test = loss_mse.detach().cpu().item()
        loss_mae = criterion_mae(y_predict, y_test).detach().cpu().item()
        loss_rmse = math.sqrt(loss_mse.detach().cpu().item())
        loss_max = MAXLoss(y_predict, y_test)
        if epoch >= 0:  
          min_avg = (min_mae[i] + min_rmse[i])/2
          loss_avg = (loss_mae + loss_rmse) / 2
          if min_avg > loss_avg:
            min_max[i] = loss_max
            min_rmse[i] = loss_rmse
            min_mae[i] = loss_mae
            test_name=source_data_path[-4:-1] + source_temp + source_test_set[i]+'best'
            plot_result(rundir,y_test, y_predict, save_image='test',test_name=test_name)
            error = 'mae = ' + str(loss_mae) + ' rmse = ' + str(loss_rmse) +  ' max = ' + str(loss_max)
            print(error)
            save_error(rundir,error,test_name,'test')
            loss_min = loss_train
            path = rundir+'/saved_model/best.pt'
            save_model(models, optimizers, loss_min, seed,path)
            print('min avg loss:{} saved model'.format(loss_avg))

        save_min(rundir,min_mae,min_rmse,min_max,epoch)
        if ((epoch+1) % eval_interval) == 0:
          test_name=source_data_path[-4:-1] + source_temp + source_test_set[i]
          plot_result(rundir,y_test, y_predict, save_image='test',test_name=test_name)
          error = 'mae = ' + str(loss_mae) + ' rmse = ' + str(loss_rmse) +  ' max = ' + str(loss_max)
          print(error)
          save_error(rundir,error,test_name,'test')
      #####
      loss_iter_domain.append(loss_train)
      loss_iter_predictor.append(loss_train_predictor)
      loss_iter_test.append(loss_test)
      loss_iter_mae.append(loss_mae)
      loss_iter_rmse.append(loss_rmse)
      loss_iter_max.append(loss_max)
      #if ( ((epoch+1) % eval_interval)==0 ) & (ifsave==True):
      #  save_model(models, optimizers, loss_min, seed, model_path='./saved_model/epoch'+str(epoch)+'.pt')
    plot_train_loss(rundir,loss_iter_domain, loss_iter_predictor, loss_iter_domain_acc, epochs)
    plot_test_loss(rundir,loss_iter_mae, loss_iter_rmse, loss_iter_max, epochs)


def pretrain_loo(rundir, source_temp, source_data_path, all_set,
                 lr=0.001, batch_size=64, epochs=2000, eval_interval=200,
                 seed=100, device_type=('cuda:0' if torch.cuda.is_available() else 'cpu'),
                 ifsave=True):
  """
  Leave-one-out pretraining as described in the paper (Section IV.D):
  9 experiments per temperature, each holding out one drive cycle for testing
  and training on the remaining 8 complete drive cycles.

  Args:
      rundir:           base name for output directories (e.g. 'pretrain_25')
      source_temp:      temperature string ('25', '10', '0', 'n10', 'n20')
      source_data_path: path to normalized_data/Pan/
      all_set:          list of 9 complete cycle filenames (Pan_all_set)
      lr:               learning rate
      batch_size, epochs, eval_interval, seed, device_type, ifsave: same as pretrain()

  Returns:
      dict with per-fold and average MAE/RMSE
  """
  fold_mae, fold_rmse = [], []

  for test_idx in range(len(all_set)):
    train_set = [all_set[i] for i in range(len(all_set)) if i != test_idx]
    test_set  = [all_set[test_idx]]
    cycle_name = all_set[test_idx].replace('.mat', '')
    fold_rundir = f'{rundir}_fold{test_idx+1}_{cycle_name}'

    print(f'\n{"="*55}')
    print(f'Fold {test_idx+1}/{len(all_set)}: test={cycle_name}  train={len(train_set)} cycles')
    print(f'{"="*55}')

    # Fresh model and optimizers for each fold
    mdls = {
      'conv': conv(), 'lstm': lstm(), 'fc': fc(), 'regression': regression(),
      'conv_s': conv(), 'lstm_s': lstm(), 'fc_s': fc(), 'regression_s': regression(),
      'discriminator': Discriminator(),
    }
    opts = {
      'conv':          optim.Adam(mdls['conv'].parameters(),          lr=lr),
      'lstm':          optim.Adam(mdls['lstm'].parameters(),          lr=lr),
      'fc':            optim.Adam(mdls['fc'].parameters(),            lr=lr),
      'regression':    optim.Adam(mdls['regression'].parameters(),    lr=lr),
      'discriminator': optim.Adam(mdls['discriminator'].parameters(), lr=lr),
    }
    criterion = nn.MSELoss(reduction='sum')

    pretrain(
      rundir=fold_rundir,
      source_temp=source_temp,
      target_temp=source_temp,
      source_data_path=source_data_path,
      source_train_set=train_set,
      source_test_set=test_set,
      models=mdls,
      criterion=criterion,
      optimizers=opts,
      batch_size=batch_size,
      epochs=epochs,
      eval_interval=eval_interval,
      seed=seed,
      device_type=device_type,
      ifsave=ifsave,
    )

    # Read best test result from min_errors file
    min_err_path = os.path.join('./run', fold_rundir, 'errors/min_errors')
    if os.path.exists(min_err_path):
      with open(min_err_path) as f:
        lines = f.readlines()
      mae_val  = ast.literal_eval(lines[0].replace('mae',  ''))[0]
      rmse_val = ast.literal_eval(lines[1].replace('rmse', ''))[0]
      fold_mae.append(mae_val)
      fold_rmse.append(rmse_val)
      print(f'Fold {test_idx+1} best: MAE={mae_val*100:.3f}%  RMSE={rmse_val*100:.3f}%')
    else:
      print(f'Fold {test_idx+1}: min_errors file not found')

  if fold_mae:
    avg_mae  = sum(fold_mae)  / len(fold_mae)
    avg_rmse = sum(fold_rmse) / len(fold_rmse)
    print(f'\n{"="*55}')
    print(f'[{source_temp}°C] Average over {len(fold_mae)} folds:')
    print(f'  MAE  = {avg_mae*100:.3f}%  (paper: ~1.09%)')
    print(f'  RMSE = {avg_rmse*100:.3f}%  (paper: ~1.44%)')
    print(f'{"="*55}')
    return {'fold_mae': fold_mae, 'fold_rmse': fold_rmse,
            'avg_mae': avg_mae, 'avg_rmse': avg_rmse}
  return {}

