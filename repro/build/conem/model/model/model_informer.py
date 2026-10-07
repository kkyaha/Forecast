import os


# [vá A4] os.chdir sang máy tác giả — đã vô hiệu

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.Embed import DataEmbedding
from model.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from model.layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from model.layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from model.layers.SelfAttention import FullAttention, ProbAttention
from model.layers.EncDecLayer import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.masking import TriangularCausalMask, ProbMask
from model.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from model.layers.SelfAttention import FullAttention, ProbAttention, AttentionLayer
from model.layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
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
                 init_weights = 5,
                 static_d_token = 30,
                 contextual_encoder_d_token = 30,
                 contextual_decoder_d_token = 30,
                 factor=1,
                distil=True):
        super(Model, self).__init__()
        self.pred_len = pred_len
        self.output_attention = output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq,
                                           dropout,len(cat_vab),len(num_vab)-1)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq,
                                           dropout,len(cat_vab),len(num_vab)-1)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc, x_dec, x_static_enc, x_static_dec, x_temporal_enc, x_temporal_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        print('Using Informer')

        #x_mark_enc = torch.cat((x_mark_enc,x_temporal_enc),dim=2)
        #x_mark_dec = torch.cat((x_mark_dec,x_temporal_dec),dim=2)
        x_mark_enc = x_temporal_enc
        x_mark_dec = x_temporal_dec

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]