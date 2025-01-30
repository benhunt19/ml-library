import pandas as pd
import numpy as np
from core.logisticRegression import *

def testLogisticRegressionBinaryUpdate() -> None:
    """
    Description:
    Load car crash survival data, remove rows with NaN, transform to binary
    
    Parameters:
    None
    """
    
    df = pd.read_csv("./data/car_crash_survival.csv")
    print("Original DataFrame:")
    print(df)
    
    cols = df.columns
    
    # Remove any row containing at least one NaN
    df_clean = df.dropna()

    y = df_clean[cols[-1]]
    x = df_clean.iloc[:, 0:-1]
    print(y)
    print(x)
            
    lr = LogisticRegression(x, y).dataToBinary()
    
def testLogisticRegressionGradientDescent() -> None:
    """
    Description:
    Perform a logistic regression using gradient descent on a dataset
    """
    
    df = pd.read_csv("./data/car_crash_survival.csv")
    print("Original DataFrame:")
    print(df)
    
    cols = df.columns

    # Remove any row containing at least one NaN
    df_clean = df.dropna()

    y = df_clean[cols[-1]]
    x = df_clean.iloc[:, 0:-1]
    print("Features")
    print(x.columns.to_numpy())
            
    lr = LogisticRegression(x, y).dataToBinary()
    alpha = 0.1
    thresh = 0.5
    lr.gradientDescent(alpha, thresh)
    
def testLogisticRegressionPlot() -> None:
    """
    Description:
    Perform a logistic regression using gradient descent on a dataset
    """
    
    df = pd.read_csv("./data/car_crash_survival.csv")
    print("Original DataFrame:")
    print(df)
    
    cols = df.columns
    # Remove any row containing at least one NaN
    df_clean = df.dropna()

    y = df_clean[cols[-1]]
    x = df_clean.iloc[:, 0:-1]
    print("Features")
    print(x.columns.to_numpy())
            
    lr = LogisticRegression(x, y).dataToBinary()
    alpha = 0.1
    thresh = 0.5
    lr.gradientDescent(alpha, thresh)
    lr.showPlot()