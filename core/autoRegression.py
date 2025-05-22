import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
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
        self.varTimeseries = None
        pass
    
    def plotTimeseries(self, alpha=None, variance=None) -> any:
        """
        Description:
        Plot timeseries data that is storred on the instance
        
        Paramaters:
        alpha (any): For use in the title
        variance (any): For use in the title
        """
        # If not a 1d timeseries
        if type(self.varTimeseries) == type(None):
            fig = plt.figure(1)
            sns.set(style="darkgrid")
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=self.timeSeries, palette="tab10", linewidth=2.5)
            plt.plot(self.timeSeries)
            plt.xlabel("Timestep")
            plt.ylabel("Value")
            title = fr"AR Simulation: $\alpha$ = {alpha}, $\sigma^2$ = {variance}" if type(variance) != type(None) and type(alpha) != type(None) else "AR Simulation"
            plt.title(title)
            plt.scatter(range(len(self.timeSeries)), self.timeSeries, color='red', s=10)
            plt.show()
        
        # If vector timeseries
        else:
            plt.title("Vector Auto Regression")
            plt.xlabel("Timestep")
            plt.ylabel("Value")
            lengendContent = []
            for index, series in enumerate(self.varTimeseries):
                plt.plot(self.varTimeseries[index])
                lengendContent.append(f"Stock {index}")
            plt.legend(lengendContent)
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
    
    def checkVARDimensions(self, coefficientMatrix: np.array, startVals: np.array):
        # Check coeff matrix has values
        if len(coefficientMatrix) == 0:
            return False

        # Check that there are 
        if type(coefficientMatrix[0]) != type(np.array([])):
            return False
        
        # Check that the start value dimensions are correct
        if len(startVals) != len(coefficientMatrix[0]):
            return False
        
        # Check that the matricies are square
        if len(coefficientMatrix[0]) != len(coefficientMatrix[0][0]):
            return False
        
        # Check all matrices are the same dinemsion
        for mat in coefficientMatrix:
            if mat.shape != coefficientMatrix[0].shape:
                return False
            
        return True
            
        
    def simulateVAR(self, coefficientMatrix: np.array, startVals: np.array, variance:float ,  timesteps=defaultTimesteps, plot=False) -> any:
        """
        Description:
        Simulate a VAR process where
        
        Paramaters:
        coefficientMatrix (list): This needs to be a list of n x n matricies
        startVals (array / floats): the start values foe each for each stock
        """
        # Check dimensionality
        assert self.checkVARDimensions(
            coefficientMatrix=coefficientMatrix,
            startVals=startVals
        ), "Please check the dimensions and values of the inputted matrix array"
        
        # Shape of each matrix, should be square
        matShape = coefficientMatrix[0].shape
        lookback = len(coefficientMatrix)
        
        # Initialise the return matrix / series
        
        self.varTimeseries = np.zeros((matShape[1], timesteps + lookback))
        
        # Create starting vals for first n elements
        # for each lookback interval
        for i, _ in enumerate(coefficientMatrix):
            # for each stock
            for j in range(matShape[1]):
                self.varTimeseries[j][i] = startVals[j]
                
        # Main loop - could be done with numpy matrix multiplication
        # for each timestep
        for t in range(timesteps):
            # for each stock
            for s in range(matShape[0]):
                # For each lookback
                stockSum = 0
                for l, _ in enumerate(coefficientMatrix):
                    # For each stock contributing to the value
                    for m in range(matShape[1]):
                        stockSum += coefficientMatrix[l][s][m] * self.varTimeseries[m][t + lookback - l]
                self.varTimeseries[s, t + lookback] = stockSum + np.random.randn() * variance
        
        # Remove initial conditions
        self.varTimeseries = self.varTimeseries[:, lookback - 1:]
        
        # Plot
        if plot:
            self.plotTimeseries()
            
        return self

        
        
        
        
    def AR1(self, timeseries: pd.Series, alpha: float, variance: float, timesteps=defaultTimesteps):
        """
        Description:
        Run AR1 on timeseries data
        
        Parameters:
        alpha (float): the rate of norm revergance
        variance (float): the variance of the noise at each timestep
        timesteps (int): number of timesteps to run for
        """
        self.timeseries = timeseries