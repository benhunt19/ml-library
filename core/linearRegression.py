import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

class LinearRegression:
    """
    Description:
    Perform linear regression on a data set with various methods
    
    Type:
    Supervised Learning
    
    Paramaters:
    data (DataFrame): the data to run regression on, there
    values (Series): the data to run regression on, there
    """
    def __init__(self, data: pd.DataFrame, values: pd.Series) -> None:
        self.data = data                    # Dataframe for data, one column per feature
        self.values = values                # Series for the avtual values
        self.X = self.data.copy(deep=True)  # Copy the data to a new dataframe
        self.X.insert(0, 'const', 1)        # Initialise the constnt component of the matrix
        self.B = []                         # Beta values for the linear relationship, f(x) = b0 + b1*x1 + b2*x2 + ...
        self.bIterations = []               # To store iterations for gradient descent
    
    def gradientDescent(self, alpha: float, iterations: int, features=None) -> None:
        """
        Perform gradient descent to minimize the cost function.

        Parameters:
        alpha (float): Learning rate.
        iterations (int): Number of iterations to run the algorithm.
        features (list): Features to include in the regression
        """
        # Rest b iterations if exists
        self.bIterations = []
        # Get the columns of the data
        features = self.X.columns if type(features) == type(None) else features # features in the  data to run off
        N = self.data.shape[0] # number of data points in set
        bStart = 1
        self.B = np.zeros(len(features)) + bStart # Initialize the betas
        self.bIterations = [self.B.copy()]
        # The algorithm aims to minimise the Residual Sum of Squares.
        # We follow gradient descent, moving each paranater in the oppisite
        # direction of the direction of the steepest slope. We then iteratively
        # move towards the minimum of the minimum of the curve (in any dimension)
        # For each iteration
        
        # Gradient descent algorithm
        for i in range(iterations):
            predictions = self.X.dot(self.B)
            errors = predictions - self.values
            for j in range(len(self.B)):
                gradient = (2/N) * (errors * self.X[features[j]]).sum()
                self.B[j] -= alpha * gradient
            self.bIterations.append(self.B.copy())
            
        # Transform to np array
        self.bIterations = np.array(self.bIterations)
        
    def analyticalSolution(self) -> None:
        """
        Description:
        This produces the analytical solution to minimise the RSS
        Of the data
        
        Parameters:
        None
        """
        npX = self.X.to_numpy()
        # add some assertions here
        # Needs to be an invertable matrix
        self.B = np.linalg.inv(npX.T.dot(npX)).dot(npX.T).dot(self.values.to_numpy())

    def showPlot(self) -> None:
        """
        Description:
        Create a 3d plot of the first two featues and the output.
        
        Parameters:
        None
        """
        # Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of the first two columns and the regression line
        cols = self.data.columns
        scatter = ax.scatter(
            self.data[cols[0]],
            self.data[cols[1]],
            self.values,
            c=self.values,
            marker='o'
        )
        legend = ["Data"]
        legendLim = 10 # Limit to number of legend entries
        
        # Get lines from the b iterationst
        minX = min(self.data[cols[0]])
        maxX = max(self.data[cols[0]])
        minY = min(self.data[cols[1]])
        maxY = max(self.data[cols[1]])
        points = 100
        xLin = np.linspace(minX, maxX, points)
        yLin = np.linspace(minY, maxY, points)
        # we are only going to plot up to 3d
        if len(self.bIterations) > 0:
            for (i, iteration) in enumerate(self.bIterations):
                zTmp = [iteration[0] + iteration[1] * xLin[i] + iteration[2] * yLin[i] for i in range(points)]
                ax.plot(xLin, yLin, zTmp)
                if len(legend) < legendLim:
                    legend.append("Iter " + str(i))
        if (len(self.B) > 0):
            z = [self.B[0] + self.B[1] * xLin[i] + self.B[2] * yLin[i] for i in range(points)]
            legend.append("Final")
            ax.plot(xLin, yLin, z)
        
        # Set labels
        ax.set_title("Liniear Regression Model")
        ax.set_xlabel(cols[0]); ax.set_ylabel(cols[1]); ax.set_zlabel("Output")
        ax.legend(legend)
        # Animation function to update view angle
        def update(frame):
            ax.view_init(elev=30, azim=frame)  # Change azimuth angle
            return ax,

        # Create animation
        self.ani =  FuncAnimation(fig, update, frames=360, interval=30, blit=False)
        
        plt.show()
    
    def saveAnimation(self, name: str) -> None:
        """
        Description:
        Save the plot animation as a gif
        
        Parameters:
        name (string): The name of the file
        """
        assert self.ani is not None
        self.ani.save(f'media/{name}.gif', writer=PillowWriter(fps=30))