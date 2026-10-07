# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 22:29:02 2022

@author: hoang.nguyen
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Auto-Fedformer
from model.layers.Embed import DataEmbedding_wo_pos, DataEmbedding_wo_decoder#, DataEmbedding
from model.layers.Correlation import AutoCorrelationLayer, MultiWaveletCross, MultiWaveletTransform
from model.layers.EncDecLayer import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi#, ConvLayer
# Transformer
#from model.layers.Transformer_EncDec import Decoder as tranDecoder, DecoderLayer as tranDecoderLayer, Encoder as tranEncoder, EncoderLayer as tranEncoderLayer, ConvLayer as tranConvLayer
#from model.layers.SelfAttention import FullAttention, AttentionLayer
#from model.layers.Embed import DataEmbedding

# Custom Embedding
from model.layers.Embed import Customize_Tokenizer
import torch.nn.init as nn_init
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPSILON = 1e-10

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
                 #static_fab_model,
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
                 init_weights = 5,
                 static_d_token = 200,
                 contextual_encoder_d_token = 200,
                 contextual_decoder_d_token = 200
                 ):
        super(Model, self).__init__()
        #self.mode_select = configs.mode_select
        self.modes = modes
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.quantiles = quantiles
        self.is_quantile = is_quantile
        #self.if_holiday_data = configs.if_holiday_data
        #self.if_promo = configs.if_promo
        self.cat_vab = cat_vab
        self.num_vab = num_vab
        self.static_vab = static_vab
        #self.static_fab_model = static_fab_model
        self.d_model = d_model
        self.RIN = RIN
        

        ## RIN
        if self.RIN:
            self.affine_weight = nn.Parameter(Tensor(1, 1, len(static_vab))) 
            self.affine_bias = nn.Parameter(Tensor(1, 1, len(static_vab)))
            nn_init.kaiming_uniform_(self.affine_weight, a=math.sqrt(init_weights))
            nn_init.kaiming_uniform_(self.affine_bias, a=math.sqrt(init_weights))
            #self.category_offsets = torch.tensor([0] + self.cat_vab[:-1]).cumsum(0)
            #self.register_buffer('category_offsets', self.category_offsets)
            #self.category_embeddings = nn.Embedding(sum(self.cat_vab), 1)
            #nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            # Static Softmax Input
            #self.linear_static_rin = nn.Linear(3,3)
            #self.static_softmax_rin = nn.Softmax(dim=2)
            
        ## Decomp
        kernel_size = moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        ## Embedding Autoformer
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq,
                                                  dropout,len(cat_vab),len(num_vab)-1)
        self.dec_embedding = DataEmbedding_wo_decoder(dec_in, d_model, embed, freq,
                                                  dropout,len(cat_vab),len(num_vab)-1)
        # Embedding Vanilla Transformer
        #self.enc_embedding_tran = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                   configs.dropout,configs.if_holiday_data, configs.if_promo, len(configs.cat_vab))
        #self.dec_embedding_tran = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
        #                                   configs.dropout,configs.if_holiday_data, configs.if_promo, len(configs.cat_vab))
        
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
            #conv_layers=[ConvLayer(
            #        d_model
            #    ) for l in range(e_layers-1)],
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
            projection=nn.Linear(d_model, c_out, bias=False) # output = trend + seasonality # NOT USING RELU/ELU FOR OUTPUT LAYER
        )
        
        #### CUSTOM EMBEDDING 
        ### TEMPORAL
        self.freq_map = {'m': 1, 'w': 3, 'd': 5}
        self.Customize_Tokenizer_Encoder_Temporal = Customize_Tokenizer(d_numerical = self.freq_map[freq], #Temporary when using label encoder
                                                               categories = None, 
                                                               d_token = contextual_encoder_d_token, 
                                                               bias = True)
        self.Customize_Tokenizer_Decoder_Temporal = Customize_Tokenizer(d_numerical = self.freq_map[freq], #Temporary when using label encoder
                                                               categories = None, 
                                                               d_token = contextual_decoder_d_token, 
                                                               bias = True)
        # Gated Linear Unit for Static Embedding
        self.linear_temporal_enc = nn.Linear(contextual_encoder_d_token,1)
        self.linear_temporal_dec = nn.Linear(contextual_decoder_d_token,1)


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
        self.linear_static = nn.Linear(static_d_token,1)
        self.linear_static_dec = nn.Linear(static_d_token,1)
        # Static Softmax Seasonality and Trend
        self.linear_context_static_seasonal_init = nn.Linear(static_d_token,1)
        self.static_softmax_seasonality = nn.Softmax(dim=2)
        # Static Softmax Input
        self.linear_context_encoder = nn.Linear(static_d_token,1) 
        self.static_softmax_encoder = nn.Softmax(dim=2)
        # Static Softmax Output
        self.linear_context_decoder = nn.Linear(static_d_token,1) 
        self.static_softmax_decoder = nn.Softmax(dim=2)


        ### TIME-VARYING
        # if temporal features are encoded by one-hot-encoding
        #self.num_cat = len(cat_vab) + self.freq_map[freq]
        #self.num_num = len(num_vab)-1
        # if temporal features are encoded as numerical values
        self.num_cat = len(cat_vab)
        self.num_num = len(num_vab)-1 #+ self.freq_map[freq] #excluding Target
        ## Feature Tokenizer
        #self.affine_weight = nn.Parameter(Tensor(1, 1, 3)) 
        #self.affine_bias = nn.Parameter(Tensor(1, 1, 3))
        #nn_init.kaiming_uniform_(self.affine_weight, a=math.sqrt(5))
        #nn_init.kaiming_uniform_(self.affine_bias, a=math.sqrt(5))
        ## Time-varying features (excluding temporal features)
        self.Customize_Tokenizer_Encoder = Customize_Tokenizer(d_numerical = self.num_num + self.num_cat + self.freq_map[freq], #Temporary when Cat encoded by label encoder
                                                               categories = None, 
                                                               d_token = contextual_encoder_d_token, 
                                                               bias = True)
        self.Customize_Tokenizer_Decoder = Customize_Tokenizer(d_numerical = self.num_num + self.num_cat + self.freq_map[freq], #Temporary when Cat encoded by label encoder
                                                               categories = None, 
                                                               d_token = contextual_decoder_d_token, 
                                                               bias = True)
       
        # Weight for Decoder in Linear Output with context embedding
        #self.enc_in_weight = nn.Parameter(Tensor(1, 8, 1))
        #nn_init.kaiming_uniform_(self.enc_in_weight, a=math.sqrt(5))
        # Weight for Decoder in Linear Output with context embedding
        #self.seasonal_init_weight = nn.Parameter(Tensor(1, 8, 1))
        #nn_init.kaiming_uniform_(self.seasonal_init_weight, a=math.sqrt(5))
        #self.dec_out_weight = nn.Parameter(Tensor(1, 13, 1))
        #nn_init.kaiming_uniform_(self.dec_out_weight, a=math.sqrt(5))
        # Number of contextual features
        self.num_context = self.num_num + self.num_cat + self.freq_map[freq]
        # Gated Linear Unit for Time-varying Embedding
        self.linear_context_enc = nn.Linear(contextual_encoder_d_token,1) #Encoder
        self.linear_context_dec = nn.Linear(contextual_decoder_d_token,1) #Decoder
        # Weight for contextual embedding with Encoder
        self.context_enc_in = nn.Parameter(Tensor(1, self.seq_len, contextual_encoder_d_token))   # encoder
        self.context_in_weight = nn.Parameter(Tensor(1, self.seq_len, contextual_encoder_d_token)) # seasonality
        nn_init.kaiming_uniform_(self.context_in_weight, a=math.sqrt(init_weights))
        # Weight for contextual embedding with Decoder
        self.context_out_weight = nn.Parameter(Tensor(1, self.label_len + self.pred_len, contextual_decoder_d_token))
        nn_init.kaiming_uniform_(self.context_out_weight, a=math.sqrt(init_weights))
        
        #### Contextual Linear


        
        #### Dropout layer
        self.dropout_dec = nn.Dropout(p=dropout)                                    
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_static_enc, x_static_dec, x_temporal_enc, x_temporal_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        ## RIN
        #print(x_enc.shape)
        #print(self.affine_weight.shape)
        #print(x_static_enc.shape)
        #print(torch.sum(self.affine_weight*x_static_enc,dim=2,keepdim=True).shape)
        if self.RIN:
            #temp = torch.zeros((1,1,1,4655))
            #for i in self.static_vab:
            #    temp = torch.zeros((1,1,1,4655))
            #print('static shape ',x_static_enc.shape)    
            #print('x in shape ',x_enc.shape)    
            means = x_enc.mean(1, keepdim=True).detach()
            #mean
            x_enc = x_enc - means
            #var
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            # affine
            # print(x.shape,self.affine_weight.shape,self.affine_bias.shape)
            #x_enc = x_enc * torch.sum(self.affine_weight*self.static_softmax_rin(self.linear_static_rin(x_static_enc)),dim=2,keepdim=True) + torch.sum(self.affine_bias*self.static_softmax_rin(self.linear_static_rin(x_static_enc)),dim=2,keepdim=True)
            x_enc = x_enc * torch.sum(self.affine_weight*x_static_enc,dim=2,keepdim=True) + torch.sum(self.affine_bias*x_static_enc,dim=2,keepdim=True)

        
        ## Temporal Embedding
        #x_temporal_enc = self.Customize_Tokenizer_Encoder_Temporal(x_num=x_temporal_enc,x_cat=None)
        #x_temporal_dec = self.Customize_Tokenizer_Decoder_Temporal(x_num=x_temporal_dec,x_cat=None)
        #temporal_enc = self.linear_temporal_enc(x_temporal_enc)*self.linear_temporal_enc(x_temporal_enc).sigmoid()
        #temporal_dec = self.linear_temporal_dec(x_temporal_dec)*self.linear_temporal_dec(x_temporal_dec).sigmoid()


        ## Static Embedding
        #x_enc_static = self.Customize_Tokenizer_static_enc(x_num=x_static_enc,x_cat=None)
        #x_dec_static = self.Customize_Tokenizer_static_dec(x_num=x_static_dec,x_cat=None)
        #static_context_enc = self.linear_static(x_enc_static)*self.linear_static(x_enc_static).sigmoid()
        #static_context_dec = self.linear_static_dec(x_dec_static)*self.linear_static_dec(x_dec_static).sigmoid()
        #print(x_enc_static.shape)
        #print(static_context_enc.shape)
        ## Time-varying Contextual Embedding
        x_mark_enc = torch.cat((x_mark_enc,x_temporal_enc),dim=2)
        x_mark_dec = torch.cat((x_mark_dec,x_temporal_dec),dim=2)
        #contextual_encoder = self.Customize_Tokenizer_Encoder(x_mark_enc,x_cat=None) 
        #contextual_decoder = self.Customize_Tokenizer_Decoder(x_mark_dec,x_cat=None)
        # contextual embedding after Gated Linear Unit
        #context_in = self.linear_context_enc(contextual_encoder)*self.linear_context_enc(contextual_encoder).sigmoid()

        #print(contextual_encoder.shape)
        #print(contextual_decoder.shape)
        #print(enc_out.shape)
        

        #### Model
        ## decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        #zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc) #res, moving_mean
        #print(seasonal_init.shape)
        #print(trend_init.shape)
        # add time-varying contextual embedding
        #temp = torch.sum(contextual_encoder*context_in*self.context_in_weight,dim=2,keepdim=True) 
        #seasonal_init = seasonal_init - temp#*self.seasonal_init_weight
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
        #enc_out = enc_out - torch.sum(contextual_encoder*context_in*self.context_enc_in,dim=2,keepdim=True) #*self.enc_in_weight
        # add static contextual embedding
        #enc_out = enc_out*self.static_softmax_encoder(self.linear_context_encoder(x_enc_static*static_context_enc))
        # encoder layers
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec,x_static_enc)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part #+ self.promo(x_static_dec)
        #dec_out = dec_out[:, -self.pred_len:, :]
        
        ## Add contextual embedding factor for decoder output
        # contextual embedding Gated Linear Unit
        #context_out = self.linear_context_dec(contextual_decoder)*self.linear_context_dec(contextual_decoder).sigmoid()
        # add time-varying contextual embedding
        #dec_out = dec_out + torch.sum(contextual_decoder*context_out*self.context_out_weight,dim=2,keepdim=True) #*self.dec_out_weight
        #dec_out = dec_out*self.static_softmax_decoder(self.linear_context_decoder(x_dec_static*static_context_dec))
        # convert dec out for final forecast
        dec_out = dec_out[:, -self.pred_len:, :]
        dec_out = self.dropout_dec(dec_out)

        print('ADD TO X_MARK')
        
        # RIN
        if self.RIN:
                dec_out = dec_out - torch.sum(self.affine_bias*x_static_dec[:, -self.pred_len:, :],dim=2,keepdim=True)
                dec_out = dec_out / (torch.sum(self.affine_weight*x_static_dec[:, -self.pred_len:, :],dim=2,keepdim=True) + 1e-10)
                dec_out = dec_out * stdev
                dec_out = dec_out + means
        
        #dec_out = self.final_out(dec_out)
        
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out
        
        #if self.output_attention:
        #    return torch.round(dec_out[:, -self.pred_len:, :]), attns ## FINAL OUTPUT, MIN = 0 (as if default = not NORMALIZE data)
        #else:
        #    return torch.round(dec_out[:, -self.pred_len:, :]))  # [B, L, D] ## FINAL OUTPUT, MIN = 0 (as if default = not NORMALIZE data)
