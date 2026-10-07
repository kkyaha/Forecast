# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:10:14 2022

@author: nhngu
"""

import numpy as np
import torch
import torch.nn as nn 

EPSILON = 1e-10

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def R_Squared(pred,true):
    SS_res = np.sum(np.square(true-pred ))
    SS_tot = np.sum(np.square(true - np.mean(true)))
    return ( 1 - SS_res/(SS_tot + np.finfo(float).eps))

def MAAPE(pred, true):
    """
    Mean Arctangent Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.arctan(np.abs((true - pred) / (true + EPSILON))))

def SMAPE(pred,true):
    return np.mean(np.abs(pred - true) / ((np.abs(pred) + np.abs(true))/2))

def RMSLE(pred,true):
    square_error = np.square((np.log(true + 1) - np.log(pred + 1)))
    mean_square_log_error = np.mean(square_error)
    rmsle_loss = np.sqrt(mean_square_log_error)
    return rmsle_loss

# Mean Absolute Logarithmic Error (MALE) 
def MALE(pred,true):
    square_error = np.abs((np.log(true + 1) - np.log(pred + 1)))
    mean_square_log_error = np.mean(square_error)
    male_loss = np.sqrt(mean_square_log_error)
    return male_loss

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    #mspe = MSPE(pred, true)
    maape = MAAPE(pred,true)
    #rsquared = R_Squared(pred,true)
    #smape = SMAPE(pred,true)
    rmsle = RMSLE(pred,true)
    male = MALE(pred,true)
    return mae, mse, rmse, mape, maape, rmsle, male

## CUSTOMIZE MAAPE LOSS FUNCTION
class Custom_MAAPE(nn.Module):
    def __init__(self):
        super(Custom_MAAPE, self).__init__();
        
    def forward(self, predictions, target):
        diff = target - predictions
        abs_perc = torch.absolute(diff / (target + EPSILON))
        arctan_result = torch.arctan(abs_perc)
        loss_value = torch.mean(arctan_result)
        return loss_value
    
## CUSTOMIZE QUANTILE LOSS FUNCTION
class QuantileLoss(nn.Module):
    ## From: https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629
    def __init__(self, quantiles):
        ##takes a list of quantiles
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, predictions, target):
        assert not target.requires_grad
        assert predictions.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            #----- adjust the Weight for Quantile if want to 
            errors = target[:,:,0] - predictions[:,:, i]
            losses.append(
                torch.max(
                   (q-1) * errors, 
                   q * errors
            ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

