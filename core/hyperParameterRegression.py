import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from core.globalFunctions import scaleData, euclidianDistance

class HyperParameterRegression:
    """
    Description:
    Performs various regressions for hyperparameter adjustments,
    this is done in the validation phase of training
    
    Type:
    Validation methods
    
    Parameters:
    None
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X              # Data points
        self.y = y              # Actual results
        self.betas= []          # Linear coeffiecients to learn
    
    def ridgeRegression(self, lmbda) -> any:
        """
        Description:
        This will take the training set, perform the
        analytical solution but introduce bias using the 
        L2-Norm
        
        Equation: (X' . X + lambda * Identity)^-1 . X' . Y
        
        Parameters:
        None
        """
        self.npX = self.X.to_numpy()
        self.npY = self.y.to_numpy()
                
        # (X' . X + lambda * Identity)^-1 . X' . Y
        start = self.npX.T.dot(self.npX) + lmbda * np.identity(len(self.npX[0]))
        self.betas = np.linalg.inv(start).dot(self.npX.T).dot(self.npY)
        print(self.betas)
        return self
    
    def testBetas(self, testX: pd.DataFrame, testY: pd.Series) -> any: 
        """
        Description:
        Once
        
        """
        assert len(self.betas) > 0, "Betas should be generated with a regression method first"
        
        print(testX.to_numpy())
        
    def showPlot(self) -> any:
        """
        Description:
        Plot the first three features from the dataset
        
        Parameters:
        None
        """
        
        # Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of the first two columns and the regression line
        legend = ["Data"]
        ax.ticklabel_format(style='plain')
        
        cols = self.X.columns
        
        ax.scatter(
            self.X[cols[0]],
            self.X[cols[1]],
            self.X[cols[2]],
            c= self.X[cols[0]],
            marker='o',
            alpha=1  # Set transparency level
        )
            
        ax.set_xlabel(cols[0]); ax.set_ylabel(cols[1]); ax.set_zlabel(cols[2])
        ax.legend(legend)
        ax.set_title('1st Thee Features')
        # Animation function to update view angle
        def update(frame):
            ax.view_init(elev=30, azim=frame)  # Change azimuth angle
            return ax,

        # Create animation
        self.ani = FuncAnimation(fig, update, frames=360, interval=15, blit=False)
        
        plt.show()
        
        return self