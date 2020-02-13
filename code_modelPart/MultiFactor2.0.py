#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 23:25:14 2020

@author: Trista FANG
"""
from scipy import io
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import h5py
import os
import scipy
from tqdm import tqdm_notebook
from tqdm import tqdm

#%% load data
def load():
    ROOT = "."
    #savingROOT = './outputData/'
    styleFactorDir = ROOT + "/Data/styleFactor_20200111.mat"
    industryFactorDir = ROOT + "/Data/industryFactor_20200111.mat"
    alphaFactorDir = ROOT + "/Data/Orth_orthFactor_20200112.mat"
    returnDir = ROOT + "/Data/returnMatrix.mat"
    stockScreenDir = ROOT + "/Data/stockScreen_20200111.mat"
    
    alphaFactorMat = h5py.File(alphaFactorDir, 'r')
    print(alphaFactorMat.keys())
    alphaFactorCube = np.transpose(alphaFactorMat['exposure'])
    print('alphaFactorCube:', alphaFactorCube.shape)
    
    styleFactorMat = h5py.File(styleFactorDir, 'r')
    print(styleFactorMat.keys())
    styleFactorCube = np.transpose(styleFactorMat['exposure'])
    print('styleFactorCube:', styleFactorCube.shape)
    
    industryFactorMat = h5py.File(industryFactorDir, 'r')
    print(industryFactorMat.keys())
    industryFactorCube = np.transpose(industryFactorMat['exposure'])
    print('industryFactorCube:', industryFactorCube.shape)
    
    returnMat = io.loadmat(returnDir)
    print(returnMat.keys())
    rts = returnMat['rts']
    print('rts:', rts.shape)
    
    stockScreenMat = h5py.File(stockScreenDir,'r')
    print(stockScreenMat.keys())
    stockScreenTable = np.transpose(stockScreenMat['stockScreenMatrix'])
    print('stockScreenTable:', stockScreenTable.shape)
    return alphaFactorCube, styleFactorCube, industryFactorCube,rts,stockScreenTable

def getShiftedReturnTable(rts):
    d_timeShift = -1
    stockReturn = pd.DataFrame(rts)
    shiftedReturnTable = stockReturn.shift(d_timeShift)
    return shiftedReturnTable
        
def getTimesliceData(timeslice, *args):
    output = []
    for cube in args:
        if np.ndim(cube)>2:
            output.append(cube[timeslice, :, :])
        else:
            output.append(cube[timeslice, :].reshape(-1,1))
    return(output)

def modelTest(alphaFactorCube,startTime, endTime, industryFactorCube,styleFactorCube, shiftedReturnTable,stockScreenTable, noRankIC = True):
    
    if np.ndim(alphaFactorCube)==2:
        allXCount = industryFactorCube.shape[-1] + styleFactorCube.shape[-1]+1
    else:
        allXCount = industryFactorCube.shape[-1] + styleFactorCube.shape[-1] +alphaFactorCube.shape[-1]
    
    factorReturnTable = np.zeros((shiftedReturnTable.shape[0], allXCount))
    predictReturnTable = np.full((shiftedReturnTable.shape[0],shiftedReturnTable.shape[1]),np.nan)
    trueReturnTable = np.full((shiftedReturnTable.shape[0],shiftedReturnTable.shape[1]),np.nan)
    modelIC = np.zeros(shiftedReturnTable.shape[0])
    Tlen = 1
    modelQueue = deque(maxlen = Tlen)
    
    for timeslice in range(startTime,endTime):
        X_alpha,X_industry,X_style, stockScreen = getTimesliceData(timeslice,alphaFactorCube,industryFactorCube,styleFactorCube,stockScreenTable)
        y_shiftedReturn = shiftedReturnTable.iloc[timeslice]
        X_all = np.concatenate([X_industry, X_style, X_alpha], axis= 1)            
        toMask = np.concatenate([np.array(y_shiftedReturn).reshape(-1, 1), X_all],axis = 1)
        finiteIndex = np.isfinite(toMask).all(axis = 1)
        validIndex = np.logical_and(finiteIndex,stockScreen.astype(bool).reshape(-1,),dtype=bool)
        validToCal = toMask[validIndex, :]
        
        if not validIndex.any() :
            continue
        # print("is there any inf:", np.isinf(validToCal).any())
        # print("is there any nan:",np.isnan(validToCal).any())
        # print("validToCal", validToCal.shape)
        
        X = validToCal[:, 1:]
        y = validToCal[:, 0]
        # print("X of ",timeslice , X.shape)
        # print("y of ",timeslice,  y.shape)
        
        #fit today model
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        
        todayFReturn = model.coef_
        factorReturnTable[timeslice, :] = todayFReturn
        modelQueue.append(todayFReturn)
        
        if len(modelQueue) < Tlen:
            continue
        elif len(modelQueue) == Tlen:
        #get value in modelQueue to become Df,this is rollingWindow for FactorReturn
            rollFReturnTdays = pd.DataFrame()
            for i in range(Tlen):
                rollFReturnTdays[i,] = modelQueue[i]
            rollFReturn = rollFReturnTdays.mean(axis = 1)   
        
        #predict tomorrow model
        timeslice = timeslice+1
        X_alpha,X_industry,X_style, stockScreen = getTimesliceData(timeslice,alphaFactorCube,industryFactorCube,styleFactorCube,stockScreenTable)
        y_shiftedReturn = shiftedReturnTable.iloc[timeslice]
        X_all = np.concatenate([X_industry, X_style, X_alpha], axis= 1)  
        # print("shape of X:", X_all.shape, "\nshape of y:", y_shiftedReturn.shape)
         
        toMask = np.concatenate([np.array(y_shiftedReturn).reshape(-1, 1), X_all],axis = 1)
        finiteIndex = np.isfinite(toMask).all(axis = 1)
        validIndex = np.logical_and(finiteIndex,stockScreen.astype(bool).reshape(-1,),dtype=bool)
        validToCal = toMask[validIndex, :]
        
        if not validIndex.any() :
           continue
        
        # print("is there any inf:", np.isinf(validToCal).any())
        # print("is there any nan:",np.isnan(validToCal).any())
        # print("validToCal", validToCal.shape)
        
        X = validToCal[:, 1:]
        y = validToCal[:, 0]
        # print("X of ",timeslice , X.shape)
        # print("y of ",timeslice,  y.shape)
        
        predictReturn = np.dot(X,rollFReturn)
        
        if noRankIC:
            modelIC[timeslice,] = np.corrcoef(predictReturn, y)[0,1]
        else:
            coefRank, p = scipy.stats.spearmanr(predictReturn, y)
            modelIC[timeslice,] = coefRank
        
        predictReturnTable[timeslice,validIndex] = predictReturn
        trueReturnTable[timeslice,validIndex] = y
    return factorReturnTable, modelIC,predictReturnTable,trueReturnTable

def plotMultiModelIC(modelIC,startTime,endTime):
    plt.figure(figsize = (15, 6))
    plt.title('meanIC of alpha model is {}'.format(round(modelIC[startTime:endTime].mean(),6)))
    plt.plot(modelIC[startTime : endTime],'-o', ms = 3)
    plt.hlines(0, 0, endTime-startTime)
    plt.hlines(modelIC[startTime:endTime].mean(), -1, 1)
    print("modelIC mean of alpha model ", ":", modelIC[startTime:endTime].mean())
    
# define long short portfolio return
def LSPortReturn(predictReturnTable, trueReturnTable, startTime, endTime, groups = 10):
    print("Start construct long short portfolio, time period is startTime {} and endTime {}.".format(startTime,endTime))
    predictReturnTable = pd.DataFrame(predictReturnTable)
    trueReturnTable = pd.DataFrame(trueReturnTable)
    predictReturnTable = predictReturnTable.loc[(predictReturnTable.index >= startTime) & (predictReturnTable.index <= endTime)]
    trueReturnTable = trueReturnTable.loc[(trueReturnTable.index >= startTime) & (trueReturnTable.index <= endTime)]
    groupTable = np.full((predictReturnTable.shape[0],predictReturnTable.shape[1]),np.nan)
    
    predict = predictReturnTable.unstack()
    predict = pd.DataFrame(predict.reset_index())
    predict.columns=['stockNum','time','stockPredictReturn']  
    
    true = trueReturnTable.unstack()
    true = pd.DataFrame(true.reset_index())
    true.columns=['stockNum','time','stockTrueReturn']  
    
    LSTable = pd.merge(predict,true,
                       left_on = ['stockNum','time'],
                       right_on = ['stockNum','time'])
    #drop nan row
    LSTable = LSTable.dropna(axis=0,how='any')
    #group 1-10: value from small to large 
    LSTable['group'] = LSTable[LSTable.columns[2]].groupby(LSTable.time).apply(lambda x:np.ceil(x.rank()/(len(x)/groups)))
    
    #long group10 and short group1
    LSResult = LSTable[LSTable.columns[3]].groupby([LSTable.time,LSTable.group]).mean()
    LS = LSResult.reset_index() 
    LS = LS.pivot(index = 'time',columns = 'group',values = 'stockTrueReturn')
    
    LSSeries = LS.iloc[:,groups - 1] - LS.iloc[:,0]
    return LSSeries

def plotLS(LSSeries):
    nav = np.cumprod(LSSeries+1)/(LSSeries.iloc[1]+1)
    plt.figure(figsize = (8, 8))
    plt.title('LS Portfolio Series')
    plt.plot(nav,'-o', ms = 3)
    return nav

def performance(nav,LSSeries):
    N = 250
    #return 
    rety = (nav.iloc[-1]+1)/(nav.iloc[0]+1)*N/len(nav)
    #SR ratio 
    Sharp = LSSeries.mean()/(LSSeries.std())*np.sqrt(N/len(LSSeries))
    #Max DrawDown 
    DD = 1 - nav/nav.cummax()
    MDD = max(DD)
      
    print('年化收益率为:{}%'.format(round(rety*100,2)))
    print('夏普比为:',round(Sharp,2))
    print('最大回撤率为：{}%'.format(round(MDD*100,2)))
    return rety,Sharp,MDD
    
#%%singleFactor test
def singleFactorTest(indexToTest, alphaFactorCube,startTime, endTime, 
                     industryFactorCube,styleFactorCube, shiftedReturnTable,stockScreenTable, noRankIC = True, doPlot = True):
    singlefactorReturnTable, singlemodelIC,predictReturnTable,trueReturnTable = modelTest(alphaFactorCube[:,:,indexToTest],startTime, endTime, industryFactorCube,
                                                                      styleFactorCube, shiftedReturnTable,stockScreenTable,noRankIC = True)
    if doPlot:
        plt.figure(figsize = (15, 6))
        plt.title('IC of alpha number:'+str(indexToTest)+'is {}.'.format(round(singlemodelIC[startTime:endTime].mean(),6)))
        plt.plot(singlemodelIC[startTime:endTime],'-o', ms = 3)
        plt.hlines(0, 0, endTime - startTime)  
        
    print("modelIC mean of alpha index ", indexToTest, ":", singlemodelIC[startTime:endTime].mean()) 
    return singlefactorReturnTable, singlemodelIC
    
#%% all singleFactor test
def singleFactorTestAll(alphaFactorCube,startTime, endTime, 
                        industryFactorCube,styleFactorCube, shiftedReturnTable,stockScreenTable,noRankIC = True, doPlot = True):
    modelICs = {}
    factorReturnTables = {}
    alphaCount = alphaFactorCube.shape[-1]
    
    if doPlot:
        fig = plt.figure(figsize=(45, (alphaCount//3+1)*10))
         
    for i in tqdm(range(alphaCount)):
        print(i)
        singlefactorReturnTable, singlemodelIC = singleFactorTest(i, alphaFactorCube,startTime, endTime,
                                                          industryFactorCube,styleFactorCube, shiftedReturnTable,
                                                          stockScreenTable, noRankIC = True ,doPlot = False)
        modelICs.update({
                i:singlemodelIC
            })
        factorReturnTables.update({
                i:singlefactorReturnTable
            })
        if doPlot:
            plt.subplot(alphaCount//3+1, 3, i+1)
            plt.plot(singlemodelIC[startTime:endTime],'-o', ms = 3)
            plt.title('meanIC of alpha number:'+str(i)+'is {}.'.format(round(singlemodelIC[startTime:endTime].mean(),6)))
            plt.hlines(0, 0, endTime - startTime)
    return modelICs,factorReturnTables
        
    
