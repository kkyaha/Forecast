# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:27:44 2022

@author: nhngu
"""

from model.data.data_factory import data_provider
from model.exp.exp_basic import Exp_Basic
from model.model.model import Model
#from model.model.model_transformer import Model as Transformer_Model
from model.utils.tools import EarlyStopping, adjust_learning_rate, pred_quantiles
from model.utils.metrics import metric
from model.utils.metrics import Custom_MAAPE, QuantileLoss
from pickle import dump, load
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model = Model(self.args).float()

        #if self.args.use_multi_gpu and self.args.use_gpu: ---------------------------->>>>> CURRENTLY ERROR
        #    model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        if self.args.is_quantile:
            print('Loss function as Quantile')
            criterion = QuantileLoss(self.args.quantiles) #------------------------------------------------- LOSS QUANTILE
        else:
            print('Loss function as MSE')
            criterion = nn.MSELoss()  #--------------------------------------------------------------------- LOSS MSE
            #criterion = Custom_MAAPE() #------------------------------------------------------------------- LOSS MAAPE
        return criterion


    def train(self, setting):
        if self.args.if_test:
            train_data, train_loader = self._get_data(flag='train')
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')
        else:
            train_data, train_loader = self._get_data(flag='train')
            vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # Promotion
                #batch_x_mark_promo = batch_x_mark_promo.float().to(self.device)
                #batch_y_mark_promo = batch_y_mark_promo.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                #print('batch_x shape',batch_x.shape)
                #print('batch_y shape',batch_y.shape)
                #print('dec_inp',dec_inp.shape)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)                        
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:].to(self.device)
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            
            # if testing for each epoch
            if self.args.if_test:
                test_loss = self.vali(test_data, test_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            
            # if using early stopping based on valid loss
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # Promotion
                #batch_x_mark_promo = batch_x_mark_promo.float().to(self.device)
                #batch_y_mark_promo = batch_y_mark_promo.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                if math.isnan(loss):
                    print('Error: loss return nan')
                    break
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load('./checkpoints/' + setting + '/checkpoint.pth'))

        preds = []
        trues = []
        inputs = []
        future_contexts = []


        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):            
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # Promotion
                #batch_x_mark_promo = batch_x_mark_promo.float().to(self.device)
                #batch_y_mark_promo = batch_y_mark_promo.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)[0]                        
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_mark_promo, batch_y_mark_promo)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                ## Additional Background
                # Target Input
                batch_x = batch_x[:,:, f_dim:].to(self.device)
                # Context
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                ### Convert to Numpy
                pred = outputs.detach().cpu().numpy()  
                true = batch_y.detach().cpu().numpy()  
                inputs_ = batch_x.detach().cpu().numpy()
                future_context = batch_y_mark.detach().cpu().numpy()
                    
                ## Append the compilation of batches
                preds.append(pred)
                trues.append(true)
                inputs.append(inputs_)
                future_contexts.append(future_context)

        preds = np.array(preds)
        trues = np.array(trues)
        inputs = np.array(inputs)
        future_contexts = np.array(future_contexts)

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1]) 
        future_contexts = future_contexts.reshape(-1, future_contexts.shape[-2], future_contexts.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        
        #### RESULTS SAVING
        ## Folder
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        ## Re-scale target
        # if self.args.scale:
            #print('SCALE DATA')
        #self.original_scale = load(open(self.args.root_path+'scaler/'+ self.args.data_path + '/scaler.pkl', 'rb'))
        folder_path = self.args.root_path+'num_vab_scaling/' + self.args.data_path
        self.original_scale = load(open(folder_path+'/scaler.pkl', 'rb'))[self.args.target]
        
        
        for i in range(preds.shape[0]): 
            preds[i,:,:] = self.original_scale.inverse_transform(preds[i,:,:])   
            trues[i,:,:] = self.original_scale.inverse_transform(trues[i,:,:])
            inputs[i,:,:] = self.original_scale.inverse_transform(inputs[i,:,:]) 
        # remove negative value
        preds[preds<0] = 0
        trues[trues<0] = 0 
        inputs[inputs<0] = 0
        
        ## Metrics
        mae, mse, rmse, mape, maape = metric(preds, trues)
        print('mse:{}, mae:{}, mape:{}, maape:{}'.format(mse,mae,mape,maape))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, mape:{}, maape:{}'.format(mse,mae,mape,maape))
        f.write('\n')
        f.write('\n')
        f.close()
    
        ## Save values    
        np.save(folder_path + 'metrics.npy', np.array([maape, mae, mse, rmse, mape]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'input.npy', inputs)
        np.save(folder_path + 'future_contexts.npy',future_contexts)

        return

    def predict(self, setting, t_load=True):
        pred_data, pred_loader = self._get_data(flag='pred')

        if t_load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            
        contexts = []
        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                #print(batch_x.shape)
                #print(batch_x)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                batch_y_mark = batch_y_mark.to(self.device)
                context = batch_y_mark.detach().cpu().numpy()
                contexts.append(context)
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)
                
        contexts = np.array(contexts)
        contexts = contexts.reshape(-1, contexts.shape[-2], contexts.shape[-1])
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
                            
        ## RESCALE
        if self.args.scale:
            #print('SCALE DATA')
            self.original_scale = load(open(self.args.scaler_path + '/scaler.pkl', 'rb'))
            for i in range(preds.shape[0]): 
                preds[i,:,:] = self.original_scale.inverse_transform(preds[i,:,:])   
                
        ## Remove Negative value
        preds[preds<0] = 0    
        
        ## Encode Group
        self.cat_encoder = load(open(self.args.cat_encode_path+'/category_encoding.pkl', 'rb')) 
        
        
        ## Convert output to df
        df_dict = {}
        list_cols = ['Sample','Prediction']
        for cat in self.args.cat_vab:
            list_cols.append(cat)
        for cols in list_cols:
            df_dict[cols] = ['A'] * preds.shape[0]
        # Count time features
        if self.args.freq == 'd':
            time_features = 5
        elif self.args.freq == 'w':
            time_features = 3
        elif self.args.freq == 'm':
            time_features = 1
        # Input value
        for sample in range(preds.shape[0]):
            df_dict['Sample'][sample] = sample
            df_dict['Prediction'][sample] = preds[sample,:,:]
            cnt = 0
            for cat in self.args.cat_vab:
                index = cnt + time_features
                df_dict[cat][sample] = self.cat_encoder[cat].inverse_transform(contexts[sample,:,index].astype('int'))[0]
                cnt += 1
        # Convert to df
        df = pd.DataFrame(df_dict)
        
        ## Result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'real_prediction.npy', preds)
        df.to_csv(folder_path+'real_forecast_df.csv',index=False)
        
        if self.args.is_quantile:
            pred_quantiles(preds,folder_path)

        return