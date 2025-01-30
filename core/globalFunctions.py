import pandas as pd
import numpy as np
from math import exp

# GLOBAL FUNCTIONS TO USE THROUGHOUT THE LIBARY

def scaleData(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Description:
    Scales the data along by the mean and squashes in by the variance.
    This essentially normalises the data so when the nearest neighbour
    algo is run, the difference values are weighted correctly.
    
    Parameters:
    data (DataFrame):
    The data to run the scaling on. Only numerical values
    Allowed
    mean (Series):
    A series containing the mean of each column
    variance (Series):
    A series containing the variance of each column
    """
    if type(data) == type(pd.Series):
        data = data.to_frame().copy()
        
    scaledData = data.copy()
    mean = scaledData.mean(axis=0)
    variance = scaledData.var(axis=0)
    
    scaledData -= mean
    scaledData /= variance.pow(0.5)
    
    return scaledData, mean, variance

def euclidianDistance(row1: pd.Series, row2:pd.Series) -> float:
    """
    Descrption:
    Calculates the euclidian distances between rows of data
    
    Parameters:
    row1 (Series): The first row of data to compare
    row2 (Series): The second row of data to compare
    """
    sum = 0
    for i in range(len(row1)):
        sum += pow(row1.iat[i] - row2.iat[i], 2)
    return pow(sum, 0.5)
    
def sigmoid(data: np.array, beta: np.array) -> float:
    """
    Definition:
    Findoing the sigmoid of a current datapoint given a current linear coefficients
    
    Parameters:
    data (Series): A point of data, with p features
    beta (Series): The current linear p + 1 coefficients
    """
    exponent = beta.dot(data).sum()
    return 1 / (1 + exp(-exponent))