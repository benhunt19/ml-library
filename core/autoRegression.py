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
        title = fr"AR1 Simulation: $\alpha$ = {alpha}, $\sigma^2$ = {variance}" if type(variance) != type(None) and type(alpha) != type(None) else "AR1 Simulation"
        plt.title(title)
        plt.scatter(range(len(self.timeSeries)), self.timeSeries, color='red', s=10)
        plt.show()
    
    def simulateAR1(self, start: float, alpha: float, variance: float, timesteps=defaultTimesteps, plot=False):
        """
        Description:
        Simulate AR1 processes from a start price
        
        Parameters:
        start (float): The start price of the asset
        alpha (flloat): The rate at which the asset converges (or not)
        variance (float): The variance on the gaussian noise per timestep
        timesteps (int): The number of timesteps to run for 
        plot (boolean): Plot the timeseries
        """
        curVal = start
        for i in range(timesteps):
            theRand = np.random.randn()
            curVal = alpha * curVal + theRand
            self.timeSeries[i] = curVal
        
        if plot:
            self.plotTimeseries(alpha=alpha, variance=variance)
        
        
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