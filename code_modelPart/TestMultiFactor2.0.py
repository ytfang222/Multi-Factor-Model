#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 23:03:20 2020

@author: Trista FANG
"""
from MultiFactor import load, getShiftedReturnTable,modelTest,\
    LSPortReturn,plotLS,plotMultiModelIC,performance,\
    singleFactorTest,singleFactorTestAll
import os
#%% set dir
os.chdir('/Users/mac/Documents/local_PHBS_multi-factor_project/phase 2/mutiFactorTest')

#%% define some static var
START_time = 1500
END_time = 2166

#load data
alphaFactorCube, styleFactorCube, industryFactorCube,rts,stockScreenTable = load()

shiftedReturnTable = getShiftedReturnTable(rts)

#%% multiFactor Model Test
factorReturnTable, modelIC,predictReturnTable,trueReturnTable = modelTest(alphaFactorCube,START_time, END_time, industryFactorCube,
                                                                          styleFactorCube, shiftedReturnTable,stockScreenTable,
                                                                          noRankIC = True)
#%% plot multi factor model IC
plotMultiModelIC(modelIC,START_time, END_time)

LSSeries = LSPortReturn(predictReturnTable, trueReturnTable, START_time, END_time, groups = 10)
nav = plotLS(LSSeries)

#%% performance evalution
rety,Sharp,MDD = performance(nav,LSSeries)

#%% singleFactor Test: Test one alpha and plot
indexToTest = 2
singlefactorReturnTable, singlemodelIC = singleFactorTest(indexToTest, alphaFactorCube,START_time, END_time,
                                                          industryFactorCube,styleFactorCube, shiftedReturnTable,
                                                          stockScreenTable, noRankIC = True ,doPlot = True)

#%% All singleFactor Test: Test all single alpha and do subplot 
modelICs,factorReturnTables = singleFactorTestAll(alphaFactorCube,START_time, END_time,
                        industryFactorCube,styleFactorCube, shiftedReturnTable,stockScreenTable,noRankIC = True, doPlot = True)
