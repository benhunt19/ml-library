import pandas as pd
import yfinance as yf
import datetime
from core.autoRegression import *

def testSimulateAR1() -> None:
    """
    Description:
    Test a simulation of an AR1 process with varying alphas
    
    Parameters:
    None
    """
    # Testing alpha < 1
    params1 = {
        "alpha": 0.9,
        "variance": 4,
        "start": 100,
        "timesteps": 40,
        "plot": True
    }
    ar1 = AutoRegression().simulateAR1(**params1)
    
    # Testing alpha = 1 (a random walk)
    params2 = {
        "alpha": 1,
        "variance": 1,
        "start": 100,
        "timesteps": 40,
        "plot": True
    }
    ar2 = AutoRegression().simulateAR1(**params2)
    
    # Testing alpha > 1 (exponential increase)
    params3 = {
        "alpha": 1.1,
        "variance": 1,
        "start": 100,
        "timesteps": 40,
        "plot": True
    }
    ar3 = AutoRegression().simulateAR1(**params3)
    
def testAR1() -> None:
    """
    Description:
    Test the AR1 method in the class autoRegression using yfinance data
    
    Parameters:
    None
    """

    apple_ticker = "AAPl"
    startDate = datetime.datetime.now() - datetime.timedelta(days=365)
    endDate = datetime.datetime.now()
    aapl_df = yf.Ticker(apple_ticker).history(start=startDate, end=endDate)
    print(aapl_df)
    
    alpha = 0.6
    variance = 0.8
    ar1 = AutoRegression.AR1(
        alpha=alpha,
        variance=variance
    )