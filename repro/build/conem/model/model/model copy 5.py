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

# Auto-Fedformer
from model.layers.Embed import DataEmbedding_encoder, DataEmbedding_decoder
from model.layers.Correlation import AutoCorrelationLayer, MultiWaveletCross, MultiWaveletTransform
from model.layers.EncDecLayer import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi

# Custom Embedding
from model.layers.Embed import Customize_Tokenizer
import torch.nn.init as nn_init
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPSILON = 1e-10

import torch
import torch.nn as nn
import torch.nn.functional as F

# https://medium.com/deeplearningmadeeasy/glu-gated-linear-unit-21e71cd52081
class FastGLU(nn.Module):
    def __init__(self,in_size):
        super().__init__()
        self.in_size = in_size
        self.linear = nn.Linear(in_size,in_size*2)
    def forward(self,x):
        out = self.linear(x)
        return out[:,:,:self.in_size]*out[:,:,self.in_size:].sigmoid()

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
                 contextual_decoder_d_token = 30
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
        self.contextual_decoder_d_token = contextual_decoder_d_token
        self.static_d_token = static_d_token
        self.enc_in = enc_in
        self.dec_in = dec_in

        ## RIN
        if self.RIN:
            self.affine_weight_input = nn.Parameter(Tensor(1, seq_len, len(static_vab))) 
            self.affine_bias_input = nn.Parameter(Tensor(1, seq_len, len(static_vab)))
            nn_init.kaiming_uniform_(self.affine_weight_input, a=math.sqrt(init_weights))
            nn_init.kaiming_uniform_(self.affine_bias_input, a=math.sqrt(init_weights))

            self.affine_weight_output = nn.Parameter(Tensor(1, pred_len, len(static_vab))) 
            self.affine_bias_output = nn.Parameter(Tensor(1, pred_len, len(static_vab)))
            nn_init.kaiming_uniform_(self.affine_weight_output, a=math.sqrt(init_weights))
            nn_init.kaiming_uniform_(self.affine_bias_output, a=math.sqrt(init_weights))
            
        ## Decomp
        kernel_size = moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        ## Embedding Autoformer
        self.enc_embedding = DataEmbedding_encoder(enc_in, d_model, embed, freq,
                                                  dropout,len(cat_vab),len(num_vab)-1)
        self.dec_embedding = DataEmbedding_decoder(dec_in, d_model, embed, freq,
                                                  dropout,len(cat_vab),len(num_vab)-1)

        ## Self encoding by Wavelet transformer   
        encoder_self_att = MultiWaveletTransform(ich=d_model, L=L, base=base)
        decoder_self_att = MultiWaveletTransform(ich=d_model, L=L, base=base)
        decoder_cross_att = MultiWaveletCross(in_channels=d_model,
                                                  out_channels=d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=modes,
                                                  ich=d_model,
                                                  base=base,
                                                  activation=cross_activation)

        ## Encoder
        enc_modes = int(min(modes, seq_len//2))
        dec_modes = int(min(modes, (seq_len//2+pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
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
        ## Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
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
            projection=nn.Linear(d_model, c_out, bias=False)
        )
        
        #### INPUT EMBEDDING 
        self.Customize_Tokenizer_Encoder_INPUT = Customize_Tokenizer(d_numerical = enc_in, #Temporary when using label encoder
                                                               categories = None, 
                                                               d_token = contextual_encoder_d_token, 
                                                               bias = True)
        self.linear_temporal_enc_dec = nn.Linear(self.enc_in*self.seq_len,self.label_len + self.pred_len)
        self.Customize_Tokenizer_Encoder_INPUT_DEC = Customize_Tokenizer(d_numerical = 1, #Temporary when using label encoder
                                                               categories = None, 
                                                               d_token = contextual_encoder_d_token, 
                                                               bias = True)
        ##### Gated Linear Unit
        self.linear_temporal_INPUT = FastGLU(contextual_encoder_d_token) #nn.Linear(contextual_encoder_d_token,1)
        self.linear_temporal_OUTPUT = FastGLU(contextual_encoder_d_token) #nn.Linear(contextual_encoder_d_token,1)


        #### CUSTOM EMBEDDING 
        ### TEMPORAL
        self.freq_map = {'m': 1, 'w': 2, 'd': 4, 'h': 5, 't': 6}
        self.Customize_Tokenizer_Encoder_Temporal = Customize_Tokenizer(d_numerical = self.freq_map[freq], #Temporary when using label encoder
                                                               categories = None, 
                                                               d_token = contextual_encoder_d_token, 
                                                               bias = True)
        self.Customize_Tokenizer_Decoder_Temporal = Customize_Tokenizer(d_numerical = self.freq_map[freq], #Temporary when using label encoder
                                                               categories = None, 
                                                               d_token = contextual_decoder_d_token, 
                                                               bias = True)
        # Gated Linear Unit for Temporal Embedding
        self.linear_temporal_enc = FastGLU(contextual_encoder_d_token) #nn.Linear(contextual_encoder_d_token,1)
        self.linear_temporal_dec = FastGLU(contextual_decoder_d_token) #nn.Linear(contextual_decoder_d_token,1)


        ### STATIC
        ## Feature Tokenizer
        self.Customize_Tokenizer_static_enc = Customize_Tokenizer(d_numerical = len(static_vab), 
                                                               categories = None, 
                                                               d_token = static_d_token, 
                                                               bias = True)
        self.Customize_Tokenizer_static_dec = Customize_Tokenizer(d_numerical = len(static_vab), 
                                                               categories = None, 
                                                               d_token = static_d_token, 
                                                               bias = True)
        # Gated Linear Unit for Static Embedding
        self.linear_static = FastGLU(static_d_token) #nn.Linear(static_d_token,1)
        self.linear_static_dec = FastGLU(static_d_token) #nn.Linear(static_d_token,1)
        # Static Softmax Seasonality and Trend
        #self.linear_context_static_seasonal_init = nn.Linear(static_d_token,1)
        # Static Softmax Input
        #self.linear_context_encoder = nn.Linear(static_d_token,1) 
        # Static Softmax Output
        #self.linear_context_decoder = nn.Linear(static_d_token,1) 


        ### TIME-VARYING
        self.num_cat = len(cat_vab)
        self.num_num = len(num_vab)-1 #+ self.freq_map[freq] #excluding Target
        self.Customize_Tokenizer_Encoder = Customize_Tokenizer(d_numerical = self.num_num + self.num_cat, #Temporary when Cat encoded by label encoder
                                                               categories = None, 
                                                               d_token = contextual_encoder_d_token, 
                                                               bias = True)
        self.Customize_Tokenizer_Decoder = Customize_Tokenizer(d_numerical = self.num_num + self.num_cat, #Temporary when Cat encoded by label encoder
                                                               categories = None, 
                                                               d_token = contextual_decoder_d_token, 
                                                               bias = True)
       
        # Number of contextual features
        self.num_context = self.num_num + self.num_cat
        # Gated Linear Unit for Time-varying Embedding
        self.linear_context_enc = FastGLU(contextual_encoder_d_token) #nn.Linear(contextual_encoder_d_token,1) #Encoder
        self.linear_context_dec = FastGLU(contextual_decoder_d_token) #nn.Linear(contextual_decoder_d_token,1) #Decoder
        # Weight for contextual embedding with Decoder
        #self.context_out_weight = nn.Parameter(Tensor(1, self.label_len + self.pred_len, 1))
        #nn_init.kaiming_uniform_(self.context_out_weight, a=math.sqrt(init_weights))
        
        #### Contextual Linear
        self.linear_combined_enc = nn.Linear(contextual_encoder_d_token+contextual_encoder_d_token+static_d_token+contextual_encoder_d_token,
                                             contextual_encoder_d_token+contextual_encoder_d_token+static_d_token+contextual_encoder_d_token)
        self.linear_combined_dec = nn.Linear(contextual_decoder_d_token+contextual_decoder_d_token+static_d_token+contextual_decoder_d_token,
                                             contextual_decoder_d_token+contextual_decoder_d_token+static_d_token+contextual_decoder_d_token)
        self.linear_combined_combined = nn.Linear(contextual_decoder_d_token+contextual_decoder_d_token+static_d_token+contextual_decoder_d_token,
                                                  contextual_decoder_d_token+contextual_decoder_d_token+static_d_token+contextual_decoder_d_token)
        self.reduced_size_linear_combined_1 = nn.Linear((self.seq_len + self.label_len + self.pred_len)*
                                                                   (self.contextual_decoder_d_token+self.contextual_decoder_d_token+self.static_d_token+self.contextual_decoder_d_token),
                                                               (self.label_len + self.pred_len)*
                                                                   (self.contextual_decoder_d_token+self.contextual_decoder_d_token+self.static_d_token+self.contextual_decoder_d_token))
        self.reduced_size_linear_combined_2 = nn.Linear((self.label_len + self.pred_len)*
                                                                   (self.contextual_decoder_d_token+self.contextual_decoder_d_token+self.static_d_token+self.contextual_decoder_d_token),
                                                               (self.label_len + self.pred_len)*
                                                                   (self.contextual_decoder_d_token))
        self.dec_combined_contextual_GLU = nn.ReLU()
        self.reduced_size_linear_combined_3 = nn.Linear((self.label_len + self.pred_len)*(self.contextual_decoder_d_token)*2,
                                                               (self.label_len + self.pred_len)*1)
        self.dec_combined_contextual_GLU2 = nn.ReLU()
        #self.linear_combined_combined_weight = nn.Parameter(Tensor(1, self.label_len + self.pred_len, 1))
        #nn_init.kaiming_uniform_(self.linear_combined_combined_weight, a=math.sqrt(init_weights))


        #### Dropout layer
        #self.randomness = torch.tensor(np.random.normal(0, 1, dec_out[0].size()), dtype=torch.float)
        self.dropout_dec = nn.Dropout(p=dropout)                                    
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_static_enc, x_static_dec, x_temporal_enc, x_temporal_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        ## RIN
        if self.RIN:
            means = x_enc.mean(1, keepdim=True).detach()
            #mean
            #print(x_enc)
            x_enc = x_enc - means
            #var
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            #print(affine_weight)
            #print(affine_bias)
            #print(self.affine_weight_input)
            #print(x_static_enc)
            #print(torch.sum(self.affine_weight_input*x_static_enc,dim=2,keepdim=True))
            #x_static_enc = x_static_enc/torch.max(x_static_enc)
            x_enc = x_enc*(torch.add(torch.sum(self.affine_weight_input*x_static_enc,dim=2,keepdim=True),1)) + torch.sum(self.affine_bias_input*x_static_enc,dim=2,keepdim=True)
            #print(x_enc)

        ## Input Embedding
        # Encoder
        x_input_enc = self.Customize_Tokenizer_Encoder_INPUT(x_num=x_enc,x_cat=None)
        x_input_enc = self.linear_temporal_INPUT(x_input_enc) #x_input_enc*(self.linear_temporal_INPUT(x_input_enc)*self.linear_temporal_INPUT(x_input_enc).sigmoid()) # Gated Linear Unit
        # Decoder
        x_enc_dec = x_enc.reshape((-1,self.seq_len*self.enc_in))
        #print(x_enc_dec.shape)
        x_enc_dec = self.linear_temporal_enc_dec(x_enc_dec)
        #print(x_enc_dec.shape)
        #x_enc_dec = x_enc_dec.reshape((x_enc.shape[0], self.label_len + self.pred_len, self.dec_in))
        x_enc_dec = x_enc_dec.unsqueeze(2)
        #print(x_enc_dec.shape)
        #x_enc_dec = x_enc_dec.repeat(1,1,1024).reshape(16,1024,1024)
        x_enc_dec = self.Customize_Tokenizer_Encoder_INPUT_DEC(x_num=x_enc_dec,x_cat=None)
        x_enc_dec = self.linear_temporal_OUTPUT(x_enc_dec) #x_enc_dec*(self.linear_temporal_OUTPUT(x_enc_dec)*self.linear_temporal_OUTPUT(x_enc_dec).sigmoid()) # Gated Linear Unit
        ## Temporal Embedding
        #print(x_temporal_enc.shape)
        #print(x_temporal_dec.shape)
        x_temporal_enc = self.Customize_Tokenizer_Encoder_Temporal(x_num=x_temporal_enc,x_cat=None)
        x_temporal_dec = self.Customize_Tokenizer_Decoder_Temporal(x_num=x_temporal_dec,x_cat=None)
        #print(x_temporal_enc.shape)
        #print(x_temporal_dec.shape)
        #temporal_enc = self.linear_temporal_enc(x_temporal_enc)*self.linear_temporal_enc(x_temporal_enc).sigmoid()
        #temporal_dec = self.linear_temporal_dec(x_temporal_dec)*self.linear_temporal_dec(x_temporal_dec).sigmoid()
        #print(temporal_enc.shape)
        #print(temporal_dec.shape)
        x_temporal_enc = self.linear_temporal_enc(x_temporal_enc) #x_temporal_enc*temporal_enc
        x_temporal_dec = self.linear_temporal_dec(x_temporal_dec) #x_temporal_dec*temporal_dec
        #print(x_temporal_enc.shape)
        #print(x_temporal_dec.shape)

        ## Static Embedding
        x_enc_static = self.Customize_Tokenizer_static_enc(x_num=x_static_enc,x_cat=None)
        x_dec_static = self.Customize_Tokenizer_static_dec(x_num=x_static_dec,x_cat=None)
        #static_context_enc = self.linear_static(x_enc_static)*self.linear_static(x_enc_static).sigmoid() # Gated Linear Unit
        #static_context_dec = self.linear_static_dec(x_dec_static)*self.linear_static_dec(x_dec_static).sigmoid() # Gated Linear Unit
        x_enc_static = self.linear_static(x_enc_static) #x_enc_static*static_context_enc # Gated Linear Unit
        x_dec_static = self.linear_static_dec(x_dec_static) #x_dec_static*static_context_dec # Gated Linear Unit
        #print(x_enc_static.shape)
        #print(static_context_enc.shape)
        
        ## Time-varying Contextual Embedding
        contextual_encoder = self.Customize_Tokenizer_Encoder(x_mark_enc,x_cat=None) 
        contextual_decoder = self.Customize_Tokenizer_Decoder(x_mark_dec,x_cat=None)
        # contextual embedding Gated Linear Unit
        #context_in = self.linear_context_enc(contextual_encoder)*self.linear_context_enc(contextual_encoder).sigmoid()
        #context_out = self.linear_context_dec(contextual_decoder)*self.linear_context_dec(contextual_decoder).sigmoid()
        contextual_encoder = self.linear_context_enc(contextual_encoder) #contextual_encoder*context_in
        contextual_decoder = self.linear_context_dec(contextual_decoder) #contextual_decoder*context_out
        #print(contextual_encoder.shape)
        #print(contextual_decoder.shape)
        #print(enc_out.shape)
        
        #### Concatenate Static & Temporal & Time-Vayring
        #print(x_temporal_enc.shape)
        #print(x_enc_static.shape)
        #print(contextual_encoder.shape)
        combined_in = torch.cat((x_input_enc,x_temporal_enc,x_enc_static,contextual_encoder),dim=2)
        combined_in = self.linear_combined_enc(combined_in)
        #print(combined_in.shape)
        combined_out = self.linear_combined_dec(torch.cat((x_enc_dec,x_temporal_dec,x_dec_static,contextual_decoder),dim=2))
        #print(combined_out.shape)
        combined_contextual = torch.cat((combined_in,combined_out),dim=1)
        dec_combined_contextual = self.linear_combined_combined(combined_contextual)
        dec_combined_contextual = dec_combined_contextual.reshape((-1,(self.seq_len + self.label_len + self.pred_len)*
                                                                   (self.contextual_decoder_d_token+self.static_d_token+self.contextual_decoder_d_token+self.contextual_decoder_d_token)))
        #print(dec_combined_contextual.shape)
        dec_combined_contextual = self.reduced_size_linear_combined_1(dec_combined_contextual)
        #print(dec_combined_contextual.shape)
        dec_combined_contextual = self.reduced_size_linear_combined_2(dec_combined_contextual)
        dec_combined_contextual = self.dec_combined_contextual_GLU(dec_combined_contextual)
        # Resize Output Embedding
        contextual_decoder_reshaped = contextual_decoder.reshape((-1,(self.label_len + self.pred_len)*self.contextual_decoder_d_token))
        # Concat Contextual Embedding with Output Embedding
        dec_combined_contextual = torch.cat((dec_combined_contextual,contextual_decoder_reshaped),dim=1)
        dec_combined_contextual = self.reduced_size_linear_combined_3(dec_combined_contextual)
        dec_combined_contextual = self.dec_combined_contextual_GLU2(dec_combined_contextual)
        dec_combined_contextual = dec_combined_contextual.unsqueeze(2)
        #print(dec_combined_contextual.shape)

        #### Model
        ## decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc) #res, moving_mean
        #print(seasonal_init.shape)
        #print(trend_init.shape)
        # add time-varying contextual embedding
        #seasonal_init = seasonal_init - torch.sum(contextual_encoder*context_in*self.context_in_weight,dim=2,keepdim=True) #*self.seasonal_init_weight
        #seasonal_init = seasonal_init - combined_in*self.context_in_weight
        seasonal_init = seasonal_init - torch.sum(combined_in,dim=2,keepdim=True)
        x_enc = x_enc - torch.sum(combined_in,dim=2,keepdim=True)
        #print("USING COMBINE IN")
        # add static contextual embedding
        #seasonal_init = seasonal_init*self.static_softmax_seasonality(self.linear_context_static_seasonal_init(x_enc_static*static_context_enc))
        #print(seasonal_init.shape)
        #trend_init = trend_init*self.static_softmax_seasonality(self.linear_context_static_seasonal_init(x_enc_static*static_context_enc))
        #print(trend_init.shape)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        ## encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc,x_static_enc)
        # add time-varying contextual embedding
        # enc_out = enc_out - torch.sum(contextual_encoder*context_in*self.context_enc_in,dim=2,keepdim=True) #*self.enc_in_weight
        # add static contextual embedding
        # enc_out = enc_out*self.static_softmax_encoder(self.linear_context_encoder(x_enc_static*static_context_enc))
        # encoder layers
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec,x_static_enc)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        #print(trend_part.shape)
        dec_out = trend_part + seasonal_part + dec_combined_contextual #+ self.promo(x_static_dec) 
        #dec_out = dec_out[:, -self.pred_len:, :]
        
        ## Add contextual embedding factor for decoder output
        # add time-varying contextual embedding
        #dec_out = dec_out + torch.sum(contextual_decoder*context_out*self.context_out_weight,dim=2,keepdim=True) #*self.dec_out_weight
        #dec_out = dec_out*self.static_softmax_decoder(self.linear_context_decoder(x_dec_static*static_context_dec))
        # convert dec out for final forecast
        dec_out = dec_out[:, -self.pred_len:, :]
        dec_out = self.dropout_dec(dec_out)
        
        print('NEW METHOD WITH X_ENC EXECUTED')

        # RIN
        if self.RIN:
                #means = dec_out.mean(1, keepdim=True).detach()
                #stdev = torch.sqrt(torch.var(dec_out, dim=1, keepdim=True, unbiased=False) + 1e-5)
                
                # Final Dec out
                #x_static_dec = x_static_dec/torch.max(x_static_dec)
                dec_out = dec_out - torch.sum(self.affine_bias_output*x_static_dec[:, -self.pred_len:, :],dim=2,keepdim=True)
                dec_out = dec_out/torch.add(torch.sum(self.affine_weight_output*x_static_dec[:, -self.pred_len:, :],dim=2,keepdim=True),1)
                dec_out = dec_out * stdev
                dec_out = dec_out + means
                print('SCALE')
        
        #dec_out = self.final_out(dec_out)
        
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out
        
        #if self.output_attention:
        #    return torch.round(dec_out[:, -self.pred_len:, :]), attns ## FINAL OUTPUT, MIN = 0 (as if default = not NORMALIZE data)
        #else:
        #    return torch.round(dec_out[:, -self.pred_len:, :]))  # [B, L, D] ## FINAL OUTPUT, MIN = 0 (as if default = not NORMALIZE data)
