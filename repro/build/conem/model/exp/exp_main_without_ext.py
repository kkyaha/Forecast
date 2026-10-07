# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 11:58:08 2022

@author: hoang.nguyen
"""

from model.data.data_factory_without_ext import data_provider
from model.exp.exp_basic import Exp_Basic
# [vá C1] Model được import động trong _build_model theo args.model_file
from model.utils.tools import EarlyStopping, adjust_learning_rate, pred_quantiles, visual, visualize_individual, save_image
from model.utils.metrics import metric
from model.utils.metrics import Custom_MAAPE, QuantileLoss
from model.utils.tools import count_parameters
from pickle import dump, load
import math
import pandas as pd
import numpy as np
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
        from importlib import import_module
        Model = import_module('model.model.' + self.args.model_file).Model
        model = Model(modes=self.args.modes,
                      seq_len=self.args.seq_len,
                      label_len=self.args.label_len,
                      pred_len=self.args.pred_len,
                      output_attention=self.args.output_attention,
                      quantiles=self.args.quantiles,
                      is_quantile=self.args.is_quantile,
                      d_model=self.args.d_model,
                      moving_avg=self.args.moving_avg,
                      enc_in=self.args.enc_in,
                      dec_in=self.args.dec_in,
                      embed=self.args.embed,
                      freq=self.args.freq,
                      dropout=self.args.dropout,
                      cat_vab=self.args.cat_vab,
                      num_vab=self.args.num_vab,
                      static_vab=self.args.static_cat_vab,
                      L=self.args.L,
                      base=self.args.base,
                      cross_activation=self.args.cross_activation,
                      n_heads=self.args.n_heads,
                      d_ff=self.args.d_ff,
                      activation=self.args.activation,
                      e_layers=self.args.e_layers,
                      c_out=self.args.c_out,
                      d_layers=self.args.d_layers,
                      RIN=self.args.RIN,
                      init_weights = self.args.init_weights,
                      static_d_token = self.args.static_d_token,
                      contextual_encoder_d_token = self.args.contextual_encoder_d_token,
                      contextual_decoder_d_token = self.args.contextual_decoder_d_token
                      ).float()

        if self.args.use_multi_gpu and self.args.use_gpu: # ->>>>> CURRENTLY ERROR WHEN USING MULTIPLE GPUs?
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
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

    #def _select_criterion_context_embedding(self):
    #    print('Loss function as MSE for Contextual Embedding')
    #    criterion_ce = nn.MSELoss()  #--------------------------------------------------------------------- LOSS MSE
    #    return criterion_ce


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
        #criterion_ce = self._select_criterion_context_embedding()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # [vá C5 — THÊM BỞI BẢN TÁI HIỆN, không có trong repo tác giả]
        # Resume cho session bị cắt giữa đường (Colab free). Mặc định TẮT.
        _start_epoch = 0
        _res_path = path + '/' + 'resume.pt'
        if getattr(self.args, 'resume', 0) and os.path.exists(_res_path):
            # weights_only=False: cần optimizer state + numpy RNG, torch>=2.6 mặc định True
            _st = torch.load(_res_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(_st['model'])
            model_optim.load_state_dict(_st['optim'])
            early_stopping.counter = _st['es_counter']
            early_stopping.best_score = _st['es_best_score']
            early_stopping.val_loss_min = _st['es_val_loss_min']
            early_stopping.early_stop = _st['es_early_stop']
            _start_epoch = _st['next_epoch']
            try:
                torch.set_rng_state(_st['rng_torch'].cpu().to(torch.uint8))
                np.random.set_state(_st['rng_numpy'])
            except Exception as _e:
                print('[C5] khong khoi phuc duoc RNG:', _e)
            if early_stopping.early_stop:
                print('[C5] run nay da early-stop truoc do, bo qua train')
                _start_epoch = self.args.train_epochs
            else:
                print('[C5] resume tu epoch %d | best vali %.7f | es_counter %d'
                      % (_start_epoch + 1, early_stopping.val_loss_min,
                         early_stopping.counter))

        for epoch in range(_start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            
            print('Number of parameters of mode: ',count_parameters(self.model))
            
            epoch_time = time.time()
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            for i, (batch_x, batch_y, batch_x_static, batch_y_static, batch_x_temporal, batch_y_temporal) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # Time-varying
                #batch_x_mark = batch_x_mark.float().to(self.device)
                #batch_y_mark = batch_y_mark.float().to(self.device)
                # Static
                batch_x_static = batch_x_static.float().to(self.device)
                batch_y_static = batch_y_static.float().to(self.device)
                # Temporal
                batch_x_temporal = batch_x_temporal.float().to(self.device)
                batch_y_temporal = batch_y_temporal.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                #print('batch_x shape',batch_x.shape)
                #print('batch_y shape',batch_y.shape)
                #print('dec_inp',dec_inp.shape)
                #print(batch_x_temporal.shape)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            outputs = self.model(batch_x, dec_inp, batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal)[0]
                        else:
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            outputs = self.model(batch_x, dec_inp, batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        outputs = self.model(batch_x, dec_inp, batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal)[0]
                    else:
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)                        
                        outputs = self.model(batch_x, dec_inp, batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal)

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
                    # [vá C4 — THÊM BỞI BẢN TÁI HIỆN, không có trong repo tác giả]
                    # 86,7% tham số là số phức nên clip_grad_norm_ không dùng được;
                    # vector_norm trên view .real/.imag tránh tensor tạm cỡ lớn.
                    _cg = getattr(self.args, 'clip_grad', 0)
                    if _cg and _cg > 0:
                        _sq = None
                        for _q in self.model.parameters():
                            if _q.grad is None:
                                continue
                            _g = _q.grad
                            if _g.is_complex():
                                _n2 = (torch.linalg.vector_norm(_g.real) ** 2
                                       + torch.linalg.vector_norm(_g.imag) ** 2)
                            else:
                                _n2 = torch.linalg.vector_norm(_g) ** 2
                            _sq = _n2 if _sq is None else _sq + _n2
                        if _sq is not None:
                            _co = _cg / (torch.sqrt(_sq) + 1e-6)
                            if _co < 1:
                                for _q in self.model.parameters():
                                    if _q.grad is not None:
                                        _q.grad.mul_(_co)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            
            # if testing for each epoch
            if self.args.if_test and getattr(self.args, 'eval_test_each_epoch', False):
                test_loss = self.vali(test_data, test_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            
            # if using early stopping based on valid loss
            early_stopping(vali_loss, self.model, path)
            # [vá C5] lưu trạng thái resume sau MỖI epoch, không chỉ khi vali giảm.
            # Ghi ra file tạm rồi os.replace -> không để lại file hỏng nếu bị cắt giữa lúc ghi.
            if getattr(self.args, 'resume', 0):
                _tmp = _res_path + '.tmp'
                torch.save({'next_epoch': epoch + 1,
                            'model': self.model.state_dict(),
                            'optim': model_optim.state_dict(),
                            'es_counter': early_stopping.counter,
                            'es_best_score': early_stopping.best_score,
                            'es_val_loss_min': early_stopping.val_loss_min,
                            'es_early_stop': early_stopping.early_stop,
                            'rng_torch': torch.get_rng_state(),
                            'rng_numpy': np.random.get_state()}, _tmp)
                os.replace(_tmp, _res_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path,map_location=self.device))

        return self.model
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            for i, (batch_x, batch_y, batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal) in enumerate(vali_loader):
                #print('SUCCESS')
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                #batch_x_mark = batch_x_mark.float().to(self.device)
                #batch_y_mark = batch_y_mark.float().to(self.device)
                # Static
                batch_x_static = batch_x_static.float().to(self.device)
                batch_y_static = batch_y_static.float().to(self.device)
                # Temporal
                batch_x_temporal = batch_x_temporal.float().to(self.device)
                batch_y_temporal = batch_y_temporal.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            outputs = self.model(batch_x, dec_inp, batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal)[0]
                        else:
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            outputs = self.model(batch_x, dec_inp, batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal)
                else:
                    if self.args.output_attention:
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        outputs = self.model(batch_x, dec_inp, batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal)[0]
                    else:
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = self.model(batch_x, dec_inp, batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:].to(self.device)
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
        path = os.path.join(self.args.checkpoints, setting)
        if test:
            print('loading model')
            #self.model.load_state_dict(torch.load('./checkpoints/' + setting + '/checkpoint.pth',map_location=self.device))
            self.model.load_state_dict(torch.load(path + '/checkpoint.pth'))
        preds = []
        trues = []
        inputs = []
        future_contexts = []


        self.model.eval()
        with torch.no_grad():
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):            
            for i, (batch_x, batch_y, batch_x_static, batch_y_static, batch_x_temporal, batch_y_temporal) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # Static
                batch_x_static = batch_x_static.float().to(self.device)
                batch_y_static = batch_y_static.float().to(self.device)
                # Temporal
                batch_x_temporal = batch_x_temporal.float().to(self.device)
                batch_y_temporal = batch_y_temporal.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            outputs = self.model(batch_x, dec_inp, batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal)[0]                        
                        else:
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            outputs = self.model(batch_x, dec_inp, batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal)
                else:
                    if self.args.output_attention:
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        outputs = self.model(batch_x, dec_inp, batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal)[0]
                    else:
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = self.model(batch_x, dec_inp, batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                ## Additional Background
                # Target Input
                batch_x = batch_x[:,:, f_dim:].to(self.device)
                # Context
                #batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                ### Convert to Numpy
                pred = outputs.detach().cpu().numpy()  
                true = batch_y.detach().cpu().numpy()  
                inputs_ = batch_x.detach().cpu().numpy()
                #future_context = batch_y_mark.detach().cpu().numpy()
                    
                ## Append the compilation of batches
                preds.append(pred)
                trues.append(true)
                inputs.append(inputs_)
                #future_contexts.append(future_context)

        preds = np.array(preds)
        trues = np.array(trues)
        inputs = np.array(inputs)
        #future_contexts = np.array(future_contexts)

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1]) 
        #future_contexts = future_contexts.reshape(-1, future_contexts.shape[-2], future_contexts.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        
        #### RESULTS SAVING
        ## Re-scale TARGET data
        if self.args.scale: 
            print('Scaling Data')
            folder_path = self.args.root_path+'num_vab_scaling/' + self.args.data_path
            if self.args.RIN == False:
                #print('SCALE DATA')
                print('Scaling Target')
                self.original_scale = load(open(folder_path + '/scaler.pkl', 'rb'))
                self.original_scale = load(open(folder_path+'/scaler.pkl', 'rb'))[self.args.target]
                for i in range(preds.shape[0]): 
                    preds[i,:,:] = self.original_scale.inverse_transform(preds[i,:,:])   
                    trues[i,:,:] = self.original_scale.inverse_transform(trues[i,:,:])
                    inputs[i,:,:] = self.original_scale.inverse_transform(inputs[i,:,:]) 
                # Visualize
                    #gt = np.concatenate((inputs[i,:,:], trues[i,:,:]), axis=0)
                    #pd = np.concatenate((inputs[i,:,:], preds[i,:,:]), axis=0)
                    #visualize_individual(i, gt, pd)
                #file_name = setting + '.pdf'
                #save_image(file_name)
        else:
            print('No Scaling')
        # remove negative value
        #preds[preds<0] = 0
        #trues[trues<0] = 0 
        #inputs[inputs<0] = 0
        
        ## Folder
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        ## Metrics mae, mse, rmse, mape, maape, rmsle, male

        mae, mse, rmse, mape, maape, rmsle, male = metric(preds, trues)
        print('mse:{}, rmse:{}, mae:{}, mape:{}, maape:{}, rmsle:{}, male:{}'.format(mse,rmse,mae,mape,maape,rmsle,male))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, rmse:{}, mae:{}, mape:{}, maape:{}, rmsle:{}, male:{}'.format(mse,rmse,mae,mape,maape,rmsle,male))
        f.write('\n')
        f.write('\n')
        f.close()
    
        ## Save values    
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, maape, rmsle, male]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'input.npy', inputs)
        #np.save(folder_path + 'future_contexts.npy',future_contexts)
        
        return

    def predict(self, setting, t_load=True):
        pred_data, pred_loader = self._get_data(flag='pred')

        if t_load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path,map_location=self.device))
            
        contexts = []
        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal) in enumerate(pred_loader):
                #print(batch_x.shape)
                #print(batch_x)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # Static
                batch_x_static = batch_x_static.float().to(self.device)
                batch_y_static = batch_y_static.float().to(self.device)
                # Temporal
                batch_x_temporal = batch_x_temporal.float().to(self.device)
                batch_y_temporal = batch_y_temporal.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal)[0]                        
                        else:
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal)
                else:
                    if self.args.output_attention:
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal)[0]
                    else:
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_x_static,batch_y_static, batch_x_temporal, batch_y_temporal)
                
                batch_y_static = batch_y_static.to(self.device)
                context = batch_y_static.detach().cpu().numpy()
                contexts.append(context)
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)
                
        contexts = np.array(contexts)
        contexts = contexts.reshape(-1, contexts.shape[-2], contexts.shape[-1])
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
                            
        ## RESCALE
        if self.args.scale:
        #    #print('SCALE DATA')
            self.original_scale = load(open(self.args.scaler_path + '/scaler.pkl', 'rb'))[self.args.target]
            for i in range(preds.shape[0]): 
                preds[i,:,:] = self.original_scale.inverse_transform(preds[i,:,:])   
                
        ## Remove Negative value
        preds[preds<0] = 0    
        
        ## Encode Group
        self.cat_encoder = load(open(self.args.cat_encode_path+'/static_encoding.pkl', 'rb')) 
        
        
        ## Convert output to df
        df_dict = {}
        list_cols = ['Sample','Prediction']
        for cat in self.args.static_cat_vab:
            list_cols.append(cat)
        for cols in list_cols:
            df_dict[cols] = ['A'] * preds.shape[0]
        # Count time features
        #if self.args.freq == 'd':
        #    time_features = 5
        #elif self.args.freq == 'w':
        #    time_features = 3
        #elif self.args.freq == 'm':
        #    time_features = 1
        # Input value
        for sample in range(preds.shape[0]):
            df_dict['Sample'][sample] = sample
            df_dict['Prediction'][sample] = preds[sample,:,:]
            cnt = 0
            for cat in self.args.static_cat_vab:
                index = cnt #+ time_features
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