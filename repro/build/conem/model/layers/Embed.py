# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 10:02:17 2022

@author: hoang.nguyen
"""

import torch
import torch.nn as nn
import math
import numpy as np

# FT Transformer
import typing as ty
from torch import Tensor
import torch.nn.init as nn_init
import torch.nn.functional as F



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, freq='h'):
        super(TemporalEmbedding, self).__init__()
        day_size = 32
        month_size = 13
        week_size = 53

        Embed = FixedEmbedding #if embed_type == 'fixed' else nn.Embedding
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
        self.week_embed = Embed(week_size, d_model)

    def forward(self, x):
        x = x.long()
        day_x = self.day_embed(x[:, :, 1])
        week_x = self.week_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return month_x + week_x #+ day_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model,freq='w',num_cat=2,num_num=0):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'m': 1, 'w': 3, 'd': 5, 'h': 5, 't': 6}
        #num_cat: number of categorical variables
        #num_num: number of numerical variables
        d_inp = freq_map[freq] + num_cat + num_num
        #self.elu = nn.ELU()
        self.embed = nn.Linear(d_inp, d_model, bias=False)
        #self.embed = nn.Embedding(d_inp,d_model).to(self.args.device)    

    def forward(self, x):
        #x = self.elu(x)
        #x = np.concatenate((x,x_vab),axis=2)
        return self.embed(x)
    
class TimeFeatureEmbedding_decoder(nn.Module):
    def __init__(self, d_model,freq='w',num_cat=2,num_num=0):
        super(TimeFeatureEmbedding_decoder, self).__init__()
        freq_map = {'m': 1, 'w': 3, 'd': 5, 'h': 5, 't': 6}
        #num_cat: number of categorical variables
        #num_num: number of numerical variables
        d_inp = freq_map[freq] #+ num_cat + num_num #- 1
        #self.elu = nn.ELU()
        self.embed = nn.Linear(d_inp, d_model, bias=False)
        #self.embed = nn.Embedding(d_inp,d_model).to(self.args.device)
        
    def forward(self, x):
        #x = self.elu(x)
        return self.embed(x)
    

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='timeF', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, 
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)
    
#class DataEmbedding_onlypos(nn.Module):
#    def __init__(self, c_in, d_model, freq='h', dropout=0.1):
#        super(DataEmbedding_onlypos, self).__init__()

#        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
#        self.position_embedding = PositionalEmbedding(d_model=d_model)
#        self.dropout = nn.Dropout(p=dropout)

#    def forward(self, x, x_mark):
#        x = self.value_embedding(x) + self.position_embedding(x)
#        return self.dropout(x)
 

#### TOKENIZER CONTEXTUAL EMBEDDING
# Data source: https://github.com/Yura52/tabular-dl-revisiting-models/blob/main/bin/ft_transformer.py
class Customize_Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            #print(f'{self.category_embeddings.weight.shape=}')

        # take [CLS] token into account
        #self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        # No [CLS]
        #self.weight = nn.Parameter(Tensor(d_numerical, d_token)) # TF Transformer
        #self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None # 
        self.weight = nn.Parameter(Tensor(1,1,d_numerical))
        self.bias = nn.Parameter(Tensor(1,1,d_bias))
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

        self.channel_out = nn.Linear(d_bias,d_token)

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        #print(x_some.shape)
        #print(x_num.shape)
        #x_num = torch.cat(
        #    [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
        #    + ([] if x_num is None else [x_num]),
        #    dim=1,
        #)
        #print('x_num ',x_num.shape)
        #print('weight ',self.weight.shape)
        #x = self.weight[None] * x_num[:, :, None]
        x = self.weight * x_num
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        #print('x ',x.shape)
        if self.bias is not None:
            # if [CLS]
            #bias = torch.cat(
            #    [
            #        torch.zeros(1, self.bias.shape[1], device=x.device),
            #        self.bias,
            #    ]
            #)
            # if NOT [CLS]
            bias = self.bias
            
            #x = x + bias[None]
            x = x + bias
        return self.channel_out(x)
    
######## Embedding Method for Contexformer  
#### ENCODER
class ShallowMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ShallowMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, context):
        x = context
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
    
class FastGLU(nn.Module):
    def __init__(self,in_size):
        super().__init__()
        self.in_size = in_size
        self.linear = nn.Linear(in_size,in_size*2)
    def forward(self,x):
        out = self.linear(x)
        return out[:,:,:self.in_size]*out[:,:,self.in_size:].sigmoid()
    

class Local_context(nn.Module):
    def __init__(self,size_in,d_model,static_d_token,num_static=1):
        super(Local_context, self).__init__()
        ### LOCAL TEMPORAL
        self.Customize_Tokenizer = Customize_Tokenizer(d_numerical = size_in, 
                                                                        categories = None, 
                                                                        d_token = static_d_token, 
                                                                        bias = True)
        self.FastGLU_temporal = FastGLU(static_d_token) 
        ### STATIC FEATURES
        self.Customize_Tokenizer_Static = Customize_Tokenizer(d_numerical = num_static, 
                                                               categories = None, 
                                                               d_token = static_d_token, 
                                                               bias = True)
        self.FastGLU_static = FastGLU(static_d_token) 
        ### COMBINE FEATURES
        self.shallow_mlp = ShallowMLP(static_d_token*2,static_d_token*2,d_model)
        self.sigmoid_layer = nn.Sigmoid()
    def forward(self,x,x_mark_static):
        x = self.FastGLU_temporal(self.Customize_Tokenizer(x,x_cat=None))
        x_mark_static = self.FastGLU_static(self.Customize_Tokenizer_Static(x_mark_static,x_cat=None))
        x_local_input = self.shallow_mlp(torch.cat((x,x_mark_static),dim=2))
        x = self.sigmoid_layer(x_local_input)
        return x, x_local_input


class Temporal_embedding(nn.Module):
    def __init__(self,d_model,d_token,freq='w',num_cat=2,num_num=0,num_static=1):
        super(Temporal_embedding, self).__init__()
        ### TEMPORAL FEATURES
        self.freq_map = {'m': 1, 'w': 3, 'd': 5, 'h': 5, 't': 6}
        self.Customize_Tokenizer_Temporal = Customize_Tokenizer(d_numerical = self.freq_map[freq], #Temporary when using label encoder
                                                               categories = None, 
                                                               d_token = d_token, 
                                                               bias = True)
        self.FastGLU_temporal = FastGLU(d_token)
        ### STATIC FEATURES
        self.Customize_Tokenizer_Static = Customize_Tokenizer(d_numerical = num_static, 
                                                               categories = None, 
                                                               d_token = d_token, 
                                                               bias = True)
        self.FastGLU_static = FastGLU(d_token) 
        ### TIME-VARYING FEATURES
        self.Customize_Tokenizer_TimeVarying = Customize_Tokenizer(d_numerical = num_num + num_cat, #Temporary when Cat encoded by label encoder
                                                               categories = None, 
                                                               d_token = d_token, 
                                                               bias = True)
        self.FastGLU_timevarying = FastGLU(d_token)

        ### COMBINE FEATURES
        self.shallow_mlp = ShallowMLP(d_token*3,d_token*3,d_model)
        
    def forward(self,x_mark,x_temporal,x_mark_static):
        #print(x_temporal.shape)
        x_temporal = self.FastGLU_temporal(self.Customize_Tokenizer_Temporal(x_temporal,x_cat=None))
        #print(x_temporal.shape)
        #print(x_mark_static.shape)
        x_mark_static = self.FastGLU_static(self.Customize_Tokenizer_Static(x_mark_static,x_cat=None))
        #print(x_mark_static.shape)
        #print(x_mark.shape)
        x_mark = self.FastGLU_timevarying(self.Customize_Tokenizer_TimeVarying(x_mark,x_cat=None))
        #print(x_mark.shape)
        x_commbined = self.shallow_mlp(torch.cat((x_temporal,x_mark_static,x_mark),dim=2))
        return x_commbined


class DataEmbedding_encoder(nn.Module):
    def __init__(self, c_in, d_model, static_d_token, contextual_encoder_d_token, embed_type='timeF',freq='h', dropout=0.1,num_cat=2,num_num=0,num_static=1):
        super(DataEmbedding_encoder, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        #self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, 
        #                                               freq=freq,
        #                                               num_cat=num_cat,
        #                                               num_num=num_num)
        self.context_embedding = Temporal_embedding(d_model=d_model,
                                                    d_token=contextual_encoder_d_token,
                                                    freq=freq,
                                                    num_cat=num_cat,
                                                    num_num=num_num,
                                                    num_static=num_static)
        self.local_context = Local_context(size_in=1,
                                           d_model=d_model,
                                           static_d_token=static_d_token,
                                            num_static=num_static)
        ### COMBINE FEATURES
        self.shallow_mlp = ShallowMLP(d_model*2,d_model*2,d_model)
        ### COMBINE WITH VALUES
        self.linear2 = nn.Linear(d_model*2,d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark,x_temporal,x_mark_static):
        #x = self.value_embedding(x) + self.temporal_embedding(x_mark) 
        x_past_context = self.context_embedding(x_mark,x_temporal,x_mark_static)
        #print(x_past_context.shape)
        x_local, x_local_input = self.local_context(x,x_mark_static)
        #print(x_local.shape)
        x_past_context = x_past_context*x_local
        #x_local_input = x_local_input*x_local
        #x = self.linear2(torch.cat([self.value_embedding(x) ,x_past_context],axis=2))
        x = self.value_embedding(x) + x_past_context + self.position_embedding(x) #+ x_local_input #
        #print('Embedding Encoder')
        return self.dropout(x), x_past_context
    
    
#### DECODER
class DataEmbedding_decoder(nn.Module):
    def __init__(self, c_in, d_model, static_d_token, contextual_decoder_d_token, embed_type='timeF',freq='h', dropout=0.1,num_cat=2,num_num=0,num_static=1):
        super(DataEmbedding_decoder, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        #self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, 
        #                                              freq=freq,
        #                                               num_cat=num_cat,
        #                                               num_num=num_num)
        self.context_embedding = Temporal_embedding(d_model=d_model,
                                                    d_token=contextual_decoder_d_token,
                                                    freq=freq,
                                                    num_cat=num_cat,
                                                    num_num=num_num,
                                                    num_static=num_static)
        self.local_context = Local_context(size_in=1,
                                           d_model=d_model,
                                           static_d_token=static_d_token,
                                            num_static=num_static)
        ### COMBINE FEATURES
        self.shallow_mlp = ShallowMLP(d_model*2,d_model*2,d_model)
        self.linear = nn.Linear(d_model*2,d_model)
        ### COMBINE WITH VALUES
        self.linear2 = nn.Linear(d_model*2,d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark_dec,x_temporal,x_mark_static,x_past_context):
        
        #x = self.value_embedding(x) + self.temporal_embedding(x_mark_dec)
        x_local, x_local_input = self.local_context(x,x_mark_static)
        x_future_context = self.context_embedding(x_mark_dec,x_temporal,x_mark_static)*x_local
        #x_future_context = x_future_context*x_local 
        
        #Combine with Past context
        #print(x_past_context.shape)
        #print(x_future_context.shape)
        x_combined = self.shallow_mlp(torch.cat([x_past_context,x_future_context],axis=2))
        x_combined = self.linear(torch.cat([x_combined,x_future_context],axis=2))
        #x = self.linear2(torch.cat([self.value_embedding(x) ,x_combined],axis=2))
        x = self.value_embedding(x) + x_combined + self.position_embedding(x)
        #print('Embedding Decoder')
        return self.dropout(x), x_combined
    

######## Embedding Method for FEDformer|Autoformer
class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='timeF', freq='h', dropout=0.1,num_cat=0,num_num=3):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding( d_model=d_model, freq=freq,num_cat=num_cat, num_num=num_num)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # try:
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        # except:
        #     a = 1
        return self.dropout(x)
    

######## Embedding Method for Informer
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='timeF', freq='h', dropout=0.1,num_cat=0,num_num=3):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding( d_model=d_model, freq=freq,num_cat=num_cat, num_num=num_num)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # try:
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        # except:
        #     a = 1
        return self.dropout(x)

    


    
    
