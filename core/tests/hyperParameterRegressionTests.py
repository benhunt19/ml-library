import pandas as pd

from core.routers.classRouter import *

def testHyperParameterRegression():
    
    dataX = pd.read_csv('data/X.csv', header=None)
    dataY = pd.read_csv('data/y.csv', header=None).squeeze()
    
    print(dataX)
    
    points, features = dataX.shape
    
    trainCount = 70
    testCount = points - trainCount
    
    trainDataXTrain = dataX.loc[0 : trainCount, :]
    trainDataYTrain = dataY.loc[0 : trainCount]
    
    lmbda = 10
    hpr = HyperParameterRegression(trainDataXTrain, trainDataYTrain).ridgeRegression(lmbda)
    # hpr.showPlot()
    
    testDataX = dataX.iloc[trainCount : -1, :]
    testDataY = dataY.iloc[trainCount : -1]
    
    print(testDataX)
    print(testDataY)
    
    hpr.testBetas(testDataX, testDataY)
    