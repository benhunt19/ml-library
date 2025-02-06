import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from core.consts import defaultTimesteps

class AutoRegression:
    """
    Description:
    Class for auto regression functions
    
    Type:
    Algorithmic Trading
    
    Parameters:
    data (series): timeseries data to fun the auto regression over
    """
    
    def __init__(self) -> any:
        self.timeSeries = pd.Series({})
        self.defaultTimesteps = defaultTimesteps
        pass
    
    def plotTimeseries(self, alpha=None, variance=None) -> any:
        """
        Description:
        Plot timeseries data that is storred on the instance
        
        Paramaters:
        alpha (any): For use in the title
        variance (any): For use in the title
        """
        fig = plt.figure(1)
        plt.plot(self.timeSeries)
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        title = fr"AR Simulation: $\alpha$ = {alpha}, $\sigma^2$ = {variance}" if type(variance) != type(None) and type(alpha) != type(None) else "AR Simulation"
        plt.title(title)
        plt.scatter(range(len(self.timeSeries)), self.timeSeries, color='red', s=10)
        plt.show()
    
    def simulateARN(self, start: float, coefficients: float, variance: float, timesteps=defaultTimesteps, plot=False):
        """
        Description:
        Simulate Auto Regression to the Nth degres from a start price and a list of coefficients
        
        Parameters:
        start (float): The start price of the asset
        coefficients (list / iterable floats): The rate at which the asset converges (or not)
        variance (float): The variance on the gaussian noise per timestep
        timesteps (int): The number of timesteps to run for 
        plot (boolean): Plot the timeseries
        """
        # Reset time timeseries incase it has been entered elsewhere
        numCoefficients = len(coefficients)
        self.timeSeries = pd.Series([start for i in range(numCoefficients)])
        for i in range(timesteps):
            nextVal = 0
            for index, coef in enumerate(coefficients):
                nextVal += self.timeSeries[i + numCoefficients - 1 - index] * coef
            nextVal += np.random.randn() * variance
            self.timeSeries.loc[len(self.timeSeries)] = nextVal
        self.timeSeries = self.timeSeries.loc[numCoefficients : ].reset_index(drop=True)
        
        if plot:
            self.plotTimeseries()
            
    def simulateAR1(self, start: float, alpha: float, variance: float, timesteps=defaultTimesteps, plot=False) -> any:
        """
        Description:
        Simulate a one step auto regressive processes from a start price
        
        Parameters:
        start (float): The start price of the asset
        alpha (float): The rate at which the asset converges (or not)
        variance (float): The variance on the gaussian noise per timestep
        timesteps (int): The number of timesteps to run for 
        plot (boolean): Plot the timeseries
        """
        # Reset time timeseries incase it has been entered elsewhere
        self.simulateARN(
            coefficients=pd.Series([alpha]),
            start=start,
            variance=variance,
            timesteps=timesteps,
            plot=plot
        )
        return self
        
    def simulateVAR(self, coefficientMatrix: np.array) -> any:
        """
        Description:
        Simulate a VAR process where
        
        Paramaters:
        coefficientMatrix (np.array): Coefficient matrix (needs to be square)
        """
        
        
        
    def AR1(self, timeseries: pd.Series, alpha: float, variance: float, timesteps=252):
        """
        Description:
        Run AR1 on timeseries data
        
        Parameters:
        alpha (float): the rate of norm revergance
        variance (float): the variance of the noise at each timestep
        timesteps (int): number of timesteps to run for
        """
        self.timeseries = timeseries