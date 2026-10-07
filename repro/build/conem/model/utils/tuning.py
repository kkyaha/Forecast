# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 23:48:36 2023

@author: nhngu
"""

from model.model.model import Model
#from model.utils.metrics import metric
from model.utils.tools import count_parameters
#import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from torch import optim
from model.data.data_loader import Dataset_Train
from torch.utils.data import DataLoader
import optuna
import os
import time


def hyperparameter(data_path,
                   root_path,
                   target,
                   time_col,
                   group_filter,
                   cat_vab,
                   cut_off,
                   num_vab,
                   static_vab, #---------ver 3
                   freq,
                   scale,
                   batch_size,
                   seq_len,
                   label_len,
                   pred_len,
                   moving_avg,
                   use_gpu=True,
                   use_multi_gpu=True,
                   d_ff = 1024,
                   gpu='0',
                   devices='0',
                   num_workers=1,
                   enc_in=1,
                   dec_in=1,
                   c_out=1,
                   modes=64, #------------------ update hyperparameter
                   L=3,#------------------ update hyperparameter
                   base='legendre',#------------------ update hyperparameter
                   cross_activation='tanh',#------------------ update hyperparameter
                   activation='relu',
                   train_epochs = 5,
                   n_trials=5,
                   # hyperparameter option
                   learning_rate=(0.0001, 0.001),
                   dropout=(0.001, 0.01),
                   #optimizer=["Adam", "RMSprop", "SGD"],
                   e_layers=(1,3),
                   d_layers=(1,3),
                   n_heads=8,#------------------ update hyperparameter
                   d_model=512,#------------------ update hyperparameter
                   RIN = True,#---------ver 3
                   init_weights = 5,#---------ver 3
                   static_d_token = 200,#---------ver 3
                   contextual_encoder_d_token = 200,#---------ver 3
                   contextual_decoder_d_token = 200#---------ver 3
                  )-> optuna.Study:

    def objective(trial: optuna.Trial) -> float:
        param = {'learning_rate': trial.suggest_loguniform('learning_rate', *learning_rate), #low-high
                 'dropout': trial.suggest_loguniform('dropout', *dropout),
                  #'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                 'e_layers' : trial.suggest_int("e_layers", *e_layers),
                'd_layers' : trial.suggest_int("d_layers", *d_layers),
                #'n_heads' : trial.suggest_int("n_heads", *n_heads),
                #'d_model' : trial.suggest_int("d_model", *d_model)
                }
        # Data Input    
        train_data = Dataset_Train(flag='train',
                        if_test=False,
                        train_perc = 0.7,
                        root_path = root_path,
                        data_path = data_path,
                        target = target,
                        time_col = time_col,
                        group_filter = group_filter,
                        cut_off = cut_off,
                        cat_vab= cat_vab, 
                        num_vab= num_vab, 
                        #static_cat_vab= ['group','store','item'], #['bio_code'] #['store','bio_code']
                        freq = freq,
                        scale = scale,
                        size=[seq_len,label_len,pred_len],

                        )
        print('Train data length: ', len(train_data))
        valid_data = Dataset_Train(flag='val',
                                if_test=False,
                                train_perc = 0.7,
                                root_path = root_path,
                                data_path = data_path,
                                target = target,
                                time_col = time_col,
                                group_filter = group_filter,
                                cut_off = cut_off,
                                cat_vab= cat_vab, 
                                num_vab=num_vab,
                                #static_cat_vab= ['group','store','item'], #['bio_code'] #['store','bio_code']
                                freq =freq,
                                scale =scale,
                                size=[seq_len,label_len,pred_len])    
        print('Valid data length: ', len(valid_data))
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False)
        valid_loader = DataLoader(
            valid_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False)

        # CPU Configuration
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu) if not use_multi_gpu else devices
            device = torch.device('cuda:{}'.format(gpu))
            print('Use GPU: cuda:{}'.format(gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')

        time_now = time.time()
        train_steps = len(train_loader)

        # Model
        model = Model(seq_len=seq_len,
                     label_len=label_len,
                     pred_len=pred_len,
                     freq=freq,
                     moving_avg=moving_avg,
                     cat_vab=cat_vab,
                     num_vab=num_vab,
                     e_layers=param['e_layers'],
                     d_layers=param['d_layers'],
                     modes=modes,
                     output_attention=False,
                     d_model=d_model,#param['d_model'],
                     enc_in=enc_in,
                     dec_in=dec_in,
                     c_out=c_out,
                     dropout=param['dropout'],
                     L=L,
                     base=base,
                     cross_activation=cross_activation,
                     n_heads=n_heads,#param['n_heads'],
                     d_ff=d_ff,
                     activation=activation).float().to(device)

        # Optimization & Loss function
        #model_optim = getattr(optim, param['optimizer'])(model.parameters(),lr= param['learning_rate'])
        model_optim = optim.Adam(model.parameters(), lr=param['learning_rate'])
        criterion = nn.MSELoss()
        
        # Training-Valid
        for epoch in range(train_epochs):
            iter_count = 0
            train_loss = []

            model.train()

            print('Number of parameters of mode: ',count_parameters(model))

            epoch_time = time.time()
            
            #Train
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
                # encoder - decoder
                #if self.args.use_amp:
                    #with torch.cuda.amp.autocast():
                        #if self.args.output_attention:
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)[0]
                        #else:
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)

                        #f_dim = -1 if self.args.features == 'MS' else 0
                        #outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        #batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        #loss = criterion(outputs, batch_y)
                        #train_loss.append(loss.item())
                #else:
                    #if self.args.output_attention:
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)[0]
                    #else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)                        
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)

                f_dim = 0 #-1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -pred_len:, f_dim:].to(device)
                batch_y = batch_y[:, -pred_len:, f_dim:].to(device)
                loss = criterion(outputs, batch_y)
                if math.isnan(loss):
                        print('Error: loss return nan')
                        break
                train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                #if self.args.use_amp:
                    #scaler.scale(loss).backward()
                    #scaler.step(model_optim)
                    #scaler.update()
                    #else:
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            
            # Valid
            total_loss = []
            model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):
                #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float()
                    batch_x_mark = batch_x_mark.float().to(device)
                    batch_y_mark = batch_y_mark.float().to(device)
                    # Promotion
                    #batch_x_mark_promo = batch_x_mark_promo.float().to(self.device)
                    #batch_y_mark_promo = batch_y_mark_promo.float().to(self.device)
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
                    # encoder - decoder
                    #if self.args.use_amp:
                        #with torch.cuda.amp.autocast():
                            #if self.args.output_attention:
                                #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)[0]
                            #else:
                                #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)
                    #else:
                        #if self.args.output_attention:
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)[0]
                        #else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)
                    f_dim = 0 #-1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -pred_len:, f_dim:]
                    batch_y = batch_y[:, -pred_len:, f_dim:].to(device)

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                    loss = criterion(pred, true)
                    if math.isnan(loss):
                        print('Error: loss return nan')
                        break
                    total_loss.append(loss)
            total_loss = np.average(total_loss)
            #self.model.train()
        return total_loss

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))