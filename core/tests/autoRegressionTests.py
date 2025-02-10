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
    
def testSimulateARN() -> None:
    """
    Description:
    Simulate Auto Regression over N amounts of steps
    
    Paramaters:
    """
    # Testing a AR(3) model
    params1 = {
        "coefficients": [0.2, 0.4, 0.3],
        "variance": 1,
        "start": 0,
        "plot": True
    }
    ar1 = AutoRegression().simulateARN(**params1)
    
    params2 = {
        "coefficients": [0.2, 0.4, 0.3, 0.1, 0.4, -0.3, -0.2],
        "variance": 1,
        "start": 10,
        "plot": True
    }
    ar2 = AutoRegression().simulateARN(**params2)
    
def testSimulateVAR() -> None:
    startVals = [
        50,
        60
    ]
    arr1 = np.array(
        [
            [0.1, 0.2],
            [0.1, 0.2]
        ]
    )
    arr2 = np.array(
        [
            [-0.1, 0.5],
            [0.15, -0.4]
        ]
    )
    # Create an array of of matricies
    coefs = [
        arr1.copy(),
        arr2.copy()
    ]
    variance = 3
    timesteps = 100
    var = AutoRegression().simulateVAR(
        coefficientMatrix=coefs,
        startVals=startVals,
        variance=variance,
        plot=True,
        timesteps=timesteps
    )
    
def testSimulateVAR3d() -> None:
    startVals = [
        50,
        60,
        100
    ]
    arr1 = np.array(
        [
            [2.8, 0.2, 0.5],
            [2.8, 0.2, 0.2],
            [2.8, 0.2, -0.1]
        ]
    )
    arr2 = np.array(
        [
            [-0.1, 0.1, 0.1],
            [0.15, -0.3, 0.3],
            [0.15, -0.2, 0.1]
        ]
    )
    # Create an array of of matricies
    coefs = [
        arr1.copy(),
        arr2.copy(),
    ]
    variance = 3
    timesteps = 100
    var = AutoRegression().simulateVAR(
        coefficientMatrix=coefs,
        startVals=startVals,
        variance=variance,
        plot=True,
        timesteps=timesteps
    )
    
def testSimulateVARrand() -> None:
    dimensions = 2
    maxPrice = 150
    startVals = np.random.rand(dimensions) * maxPrice
    arr1 = np.random.rand(dimensions, dimensions) - 1
    arr2 = np.random.rand(dimensions, dimensions) - 1
    # Create an array of of matricies
    coefs = [
        arr1.copy(),
        arr2.copy(),
    ]
    variance = 3
    timesteps = 100
    var = AutoRegression().simulateVAR(
        coefficientMatrix=coefs,
        startVals=startVals,
        variance=variance,
        plot=True,
        timesteps=timesteps
    )

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