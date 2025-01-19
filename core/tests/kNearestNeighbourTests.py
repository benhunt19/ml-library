import pandas as pd
import numpy as np
from core.kNearestNeighbours import *

def testScaleData()-> None:
    """
    Description:
    Test scaling the data by the variance and the mean
    
    Parameters:
    None
    """
    # Load data from a CSV file into a pandas DataFrame
    data = pd.read_csv('data/house_data_clean.csv')
    
    cols = pd.Index([
        'bedrooms',
        'bathrooms',
        'sqft_living',
        'sqft_lot',
        'floors'
    ])
    
    # Select certain columns from the DataFrame
    rowLim = 1000
    dataToCompare = data.loc[0:rowLim, cols].copy()
    values = data.loc[0:rowLim, 'price'].copy()
    
    KNN = KNearestNeighbours(dataToCompare, values)
    KNN.scaleData()
    
def testKNearestNeighbours()-> None:
    """
    Description:
    Test the K nearest neighbours algorithm
    
    Parameters:
    None
    """
    # Load data from a CSV file into a pandas DataFrame
    data = pd.read_csv('data/house_data_clean.csv')
    
    cols = pd.Index([
        'bedrooms',
        'bathrooms',
        'sqft_living',
        'sqft_lot',
        'floors'
    ])
    
    # Select certain columns from the DataFrame
    rowLim = 1000
    dataToCompare = data.loc[0:rowLim, cols].copy()
    values = data.loc[0:rowLim, 'price'].copy()
    KNN = KNearestNeighbours(dataToCompare, values)
    KNN.scaleData()
    
    testRow = data.loc[rowLim + 1, cols].copy()
    K = 20
    neighbours = KNN.findNeighbours(testRow.to_frame(), K)
    fromOriginal = data.loc[neighbours.index].copy()
    meanPrice = fromOriginal.loc[:, 'price'].mean()
    print(meanPrice)

def testVisuliseNeighbours() -> None:
    """
    Description: Test the show plot animation method
    
    Parameters:
    None
    """
    # Load data from a CSV file into a pandas DataFrame
    data = pd.read_csv('data/house_data_clean.csv')
    
    # Three dimensions for graphical interpretation
    cols = pd.Index([
        'sqft_lot',
        'sqft_living',
        'price',
    ])
    
    # Select certain columns from the DataFrame
    rowLim = 123
    dataToCompare = data.loc[0:rowLim, cols].copy()
    values = data.loc[0:rowLim, 'price'].copy()
    KNN = KNearestNeighbours(dataToCompare, values)
    KNN.scaleData()
    testRow = data.loc[rowLim + 1, cols].copy()
    # print('testRow', testRow)
    K = 8
    neighbours = KNN.findNeighbours(testRow, K)
    KNN.showPlot()
    # KNN.saveAnimation(name='kNearestNeighbours')