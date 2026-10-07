# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 22:29:02 2022

@author: hoang.nguyen
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# Auto-Fedformer
from model.layers.Embed import DataEmbedding_wo_pos, DataEmbedding
from model.layers.Correlation import AutoCorrelationLayer, MultiWaveletCross, MultiWaveletTransform
from model.layers.EncDecLayer import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
# Transformer
#from model.layers.Transformer_EncDec import Decoder as tranDecoder, DecoderLayer as tranDecoderLayer, Encoder as tranEncoder, EncoderLayer as tranEncoderLayer, ConvLayer as tranConvLayer
#from model.layers.SelfAttention import FullAttention, AttentionLayer
#from model.layers.Embed import DataEmbedding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.quantiles = configs.quantiles
        self.is_quantile = configs.is_quantile
        #self.if_holiday_data = configs.if_holiday_data
        #self.if_promo = configs.if_promo
        self.cat_vab = configs.cat_vab
        self.num_vab = configs.num_vab
        self.d_model = configs.d_model

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding Autoformer
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout,len(configs.cat_vab),len(configs.num_vab)-1)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout,len(configs.cat_vab),len(configs.num_vab)-1)
        # Embedding Vanilla Transformer
        #self.enc_embedding_tran = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                   configs.dropout,configs.if_holiday_data, configs.if_promo, len(configs.cat_vab))
        #self.dec_embedding_tran = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
        #                                   configs.dropout,configs.if_holiday_data, configs.if_promo, len(configs.cat_vab))
        
        # Self encoding by Wavelet transformer   
        encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
        decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
        decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=configs.modes,
                                                  ich=configs.d_model,
                                                  base=configs.base,
                                                  activation=configs.cross_activation)

        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len//2))
        dec_modes = int(min(configs.modes, (configs.seq_len//2+configs.pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=False) # output = trend + seasonality
        )
        
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, #x_promo_enc, x_promo_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        ## Auto+Fedformer
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        #zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :]#+dec_out_tran[:, -self.pred_len:, :], attns+attns_trans
        else:
            return dec_out[:, -self.pred_len:, :]#+dec_out_tran[:, -self.pred_len:, :]  # [B, L, D] 
        
        #if self.output_attention:
        #    return torch.round(F.relu(dec_out[:, -self.pred_len:, :])), attns # --------------------------------------------------- FINAL OUTPUT, MIN = 0 (as if default = not NORMALIZE data)
        #else:
        #    return torch.round(F.relu(dec_out[:, -self.pred_len:, :]))  # [B, L, D] # --------------------------------------------- FINAL OUTPUT, MIN = 0 (as if default = not NORMALIZE data)
