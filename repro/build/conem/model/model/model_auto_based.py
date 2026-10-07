# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 22:29:02 2022

@author: hoang.nguyen
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import einops

import os

#os.chdir("/home/ad/20813716/transformer_sales_forecast/model")


# Auto-Fedformer
from model.layers.Embed import DataEmbedding_encoder, DataEmbedding_decoder
#from model.layers.Correlation import AutoCorrelationLayer, MultiWaveletCross, MultiWaveletTransform
#from model.layers.EncDecLayer import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from model.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from model.layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from model.layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from model.layers.SelfAttention import FullAttention, ProbAttention
from model.layers.EncDecLayer import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
import numpy as np

# Custom Embedding
from model.layers.Embed import Customize_Tokenizer
import torch.nn.init as nn_init
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPSILON = 1e-10

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, 
                 seq_len,
                 label_len,
                 pred_len,
                 freq,
                 moving_avg,
                 cat_vab,
                 num_vab,
                 static_vab,
                 e_layers=1,
                 d_layers=1,
                 modes=64,
                 output_attention=False,
                 quantiles=[0.1,0.3,0.5,0.7,0.9],
                 is_quantile=False,
                 d_model=512,
                 enc_in=1,
                 dec_in=1,
                 c_out=1,
                 embed='timeF',
                 dropout=0.01,
                 L=3,
                 base='legendre',
                 cross_activation='tanh',
                 n_heads=8,
                 d_ff=1024,
                 activation='relu',
                 RIN = True,
                 init_weights = 1,
                 static_d_token = 30,
                 contextual_encoder_d_token = 30,
                 contextual_decoder_d_token = 30,
                 factor = 1,
                 distil = True
                 ):
        super(Model, self).__init__()
        self.modes = modes
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.quantiles = quantiles
        self.is_quantile = is_quantile
        self.cat_vab = cat_vab
        self.num_vab = num_vab
        self.static_vab = static_vab
        self.d_model = d_model
        self.RIN = RIN
        self.freq = freq
        self.contextual_encoder_d_token = contextual_encoder_d_token
        self.contextual_decoder_d_token = contextual_decoder_d_token
        self.static_d_token = static_d_token
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.factor = factor
        self.distil = distil

        ## RIN
        if self.RIN:
            self.affine_weight_input =  nn.Parameter(Tensor(1, 1, len(static_vab))) # nn.Parameter(Tensor(1, seq_len, len(static_vab))) 
            self.affine_bias_input = nn.Parameter(Tensor(1, 1, len(static_vab)))
            nn_init.kaiming_uniform_(self.affine_weight_input, a=math.sqrt(init_weights))
            nn_init.kaiming_uniform_(self.affine_bias_input, a=math.sqrt(init_weights))

            #self.affine_weight_output = nn.Parameter(Tensor(1, pred_len, len(static_vab))) 
            #self.affine_bias_output = nn.Parameter(Tensor(1, pred_len, len(static_vab)))
            #nn_init.kaiming_uniform_(self.affine_weight_output, a=math.sqrt(init_weights))
            #nn_init.kaiming_uniform_(self.affine_bias_output, a=math.sqrt(init_weights))
            
        # Decomp
        kernel_size = moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp(kernel_size[0])
        else:
            self.decomp = series_decomp(kernel_size)

        ## Embedding Autoformer
        self.enc_embedding = DataEmbedding_encoder(enc_in, d_model, static_d_token, contextual_encoder_d_token, embed, freq,
                                                  dropout,len(cat_vab),len(num_vab)-1,len(static_vab))
        self.dec_embedding = DataEmbedding_decoder(dec_in, d_model, static_d_token, contextual_decoder_d_token, embed, freq,
                                                  dropout,len(cat_vab),len(num_vab)-1,len(static_vab))

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )
        self.dropout_enc = nn.Dropout(p=dropout) 
        self.dropout_dec = nn.Dropout(p=dropout)                                    
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_static_enc, x_static_dec, x_temporal_enc, x_temporal_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        #### STATNORM
        if self.RIN:
            means = x_enc.mean(1, keepdim=True).detach()
            #mean
            #print(x_enc)
            x_enc = x_enc - means
            #var
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            x_static_enc = torch.add(x_static_enc/x_static_enc.shape[2],-0.5)
            #print('NEW REVIN')
            #x_enc = x_enc*(torch.add(torch.sum(self.affine_weight_input*x_static_enc,dim=2,keepdim=True),1)) + torch.sum(self.affine_bias_input*x_static_enc,dim=2,keepdim=True)
            x_enc = x_enc*torch.sum(self.affine_weight_input*x_static_enc,dim=2,keepdim=True) + torch.sum(self.affine_bias_input*x_static_enc,dim=2,keepdim=True)
        #### CONEM
        ## Input Embedding
        # Encoder
        #x_input_enc = self.Customize_Tokenizer_Encoder_INPUT(x_num=x_enc,x_cat=None)
        #x_input_enc = self.FastGLU_enc(x_input_enc) # Gated Linear Unit
        # Decoder
        #x_enc_dec = x_enc.reshape((-1,self.seq_len*self.enc_in))
        #x_enc_dec = self.linear_temporal_enc_dec(x_enc_dec)
        #x_enc_dec = x_enc_dec.unsqueeze(2)
        #x_enc_dec = self.Customize_Tokenizer_Encoder_INPUT_DEC(x_num=x_enc_dec,x_cat=None)
        #x_enc_dec = self.FastGLU_dec(x_enc_dec) # Gated Linear Unit
        ## Temporal Embedding
        #x_temporal_enc = self.Customize_Tokenizer_Encoder_Temporal(x_num=x_temporal_enc,x_cat=None)
        #x_temporal_dec = self.Customize_Tokenizer_Decoder_Temporal(x_num=x_temporal_dec,x_cat=None)
        #x_temporal_enc = self.FastGLU_temporal_enc(x_temporal_enc)  # Gated Linear Unit
        #x_temporal_dec = self.FastGLU_temporal_dec(x_temporal_dec)  # Gated Linear Unit
        ## Static Embedding
        #x_enc_static = self.Customize_Tokenizer_static_enc(x_num=x_static_enc,x_cat=None)
        #x_dec_static = self.Customize_Tokenizer_static_dec(x_num=x_static_dec,x_cat=None)
        #x_enc_static = self.FastGLU_static_enc(x_enc_static) # Gated Linear Unit
        #x_dec_static = self.FastGLU_static_dec(x_dec_static) # Gated Linear Unit
        ## Time-varying Contextual Embedding
        #contextual_encoder = self.Customize_Tokenizer_Encoder(x_mark_enc,x_cat=None) 
        #contextual_decoder = self.Customize_Tokenizer_Decoder(x_mark_dec,x_cat=None)
        # contextual embedding Gated Linear Unit
        #contextual_encoder = self.FastGLU_context_enc(contextual_encoder) # Gated Linear Unit
        #contextual_decoder = self.FastGLU_context_dec(contextual_decoder) # Gated Linear Unit
        
        ##Concatenate Static & Temporal & Time-Vayring
        #combined_in = torch.cat((x_input_enc,x_temporal_enc,x_enc_static,contextual_encoder),dim=2)
        #combined_in = self.linear_combined_enc(combined_in)
        #combined_out = torch.cat((x_enc_dec,x_temporal_dec,x_dec_static,contextual_decoder),dim=2)
        #combined_out = self.linear_combined_dec(combined_out)
        #combine_in + combine_out
        #combined_contextual = torch.cat((combined_in,combined_out),dim=1)
        #dec_combined_contextual = combined_contextual #self.linear_combined_combined(combined_contextual)
        #dec_combined_contextual = dec_combined_contextual.reshape((-1,(self.seq_len + self.label_len + self.pred_len)*
        #                                                           (self.contextual_decoder_d_token+self.static_d_token+self.contextual_decoder_d_token+self.contextual_decoder_d_token)))
        #dec_combined_contextual = self.reduced_size_linear_combined_1(dec_combined_contextual)
        #dec_combined_contextual = self.reduced_size_linear_combined_2(dec_combined_contextual)
        #dec_combined_contextual = self.dec_combined_contextual_GLU(dec_combined_contextual)
        # Resize Output Embedding
        #contextual_decoder_reshaped = contextual_decoder.reshape((-1,(self.label_len + self.pred_len)*self.contextual_decoder_d_token))
        # Concat Contextual Embedding with Output Embedding
        #dec_combined_contextual = torch.cat((dec_combined_contextual,contextual_decoder_reshaped),dim=1)
        #dec_combined_contextual = self.reduced_size_linear_combined_3(dec_combined_contextual)
        #dec_combined_contextual = self.dec_combined_contextual_GLU2(dec_combined_contextual)
        #dec_combined_contextual = dec_combined_contextual.unsqueeze(2)
        #print(dec_combined_contextual.shape)

        #### Model
        #x_mark_enc = torch.cat((x_mark_enc,x_temporal_enc),dim=2)
        #x_mark_dec = torch.cat((x_mark_dec,x_temporal_dec),dim=2)
        ## decomp init
        #mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        #seasonal_init, trend_init = self.decomp(x_enc) #res, moving_mean
        # add time-varying contextual embedding
        #seasonal_init = seasonal_init - torch.sum(combined_in,dim=2,keepdim=True)
        #x_enc = x_enc - torch.sum(combined_in,dim=2,keepdim=True)
        # decoder input
        #trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        #seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        ## encoder
        #enc_out,  x_past_context = self.enc_embedding(x_enc, x_mark_enc,x_temporal_enc,x_static_enc) #x_mark,x_temporal,x_mark_static
        # encoder layers
        #enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        # dec
        #if self.pred_len+self.label_len <= self.seq_len:
        #    x_past_context_init = x_past_context[:, -(self.pred_len+self.label_len):, :]
        #else:
        #    add_length = self.pred_len+self.label_len - self.seq_len
        #    if add_length <= self.seq_len:
        #        x_past_context_init = torch.cat([x_past_context,x_past_context[:, -add_length:, :]], dim=1)
        #    else:
        #        add_on = einops.repeat(x_past_context[:, -1:, :], 'b h w -> b (repeat h) w', repeat=add_length)
                #print(add_on.shape)
        #        x_past_context_init = torch.cat([x_past_context,add_on], dim=1)
                #print(x_past_context_init.shape)
        #dec_out = self.dec_embedding(seasonal_init, x_mark_dec,x_temporal_dec,x_static_dec,x_past_context_init)
        #seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
        #                                         trend=trend_init)
        #### Final Forecast
        #dec_out = trend_part + seasonal_part 
        #dec_out = self.ShallowMLP(dec_out,dec_combined_contextual)
        #convert dec out for final forecast
        #dec_out = dec_out[:, -self.pred_len:, :]
        #dec_out = self.dropout_dec(dec_out)

        #### Autoformer
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out,  x_past_context = self.enc_embedding(x_enc, x_mark_enc,x_temporal_enc,x_static_enc) 
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = self.dropout_enc(enc_out)

        if self.pred_len+self.label_len <= self.seq_len:
            x_past_context_init = x_past_context[:, -(self.pred_len+self.label_len):, :]
        else:
            add_length = self.pred_len+self.label_len - self.seq_len
            if add_length <= self.seq_len:
                x_past_context_init = torch.cat([x_past_context,x_past_context[:, -add_length:, :]], dim=1)
            else:
                add_on = einops.repeat(x_past_context[:, -1:, :], 'b h w -> b (repeat h) w', repeat=add_length)
                #print(add_on.shape)
                x_past_context_init = torch.cat([x_past_context,add_on], dim=1)
                #print(x_past_context_init.shape)


        # dec
        dec_out, x_future_context = self.dec_embedding(seasonal_init, x_mark_dec,x_temporal_dec,x_static_dec,x_past_context_init)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        dec_out = dec_out[:, -self.pred_len:, :]
        dec_out = self.dropout_dec(dec_out)
        

        print('PROPOSAL EXP WITH AUTOFORMER')

        #### RESCALED BY STATNORM
        if self.RIN:               
            # Final Dec out
            x_static_dec = torch.add(x_static_dec/x_static_dec.shape[2],-0.5)
            dec_out = dec_out - torch.sum(self.affine_bias_input*x_static_dec[:,-self.pred_len:,:],dim=2,keepdim=True)
            dec_out = dec_out/torch.sum(self.affine_weight_input*x_static_dec[:,-self.pred_len:,:],dim=2,keepdim=True)
            #dec_out = dec_out - self.affine_bias_input
            #dec_out = dec_out/self.affine_weight_input
            dec_out = dec_out * stdev
            dec_out = dec_out + means
                
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out
        
        #if self.output_attention:
        #    return torch.round(dec_out[:, -self.pred_len:, :]), attns ## FINAL OUTPUT, MIN = 0 (as if default = not NORMALIZE data)
        #else:
        #    return torch.round(dec_out[:, -self.pred_len:, :]))  # [B, L, D] ## FINAL OUTPUT, MIN = 0 (as if default = not NORMALIZE data)
