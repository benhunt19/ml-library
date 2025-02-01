import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from core.globalFunctions import sigmoid, scaleData

class LogisticRegression:
    """
    Description:
    Perform logistic regression to define hyperplanes within data sets.
    The learnt function (and corresponding betas) will be the normal vector the hyperplane
    
    Type:
    Supervised Learning
    
    Paramaters:
    data (DataFrame): the data to run regression on, there
    values (Series): the data to run regression on, there
    """
    
    def __init__(self, data: pd.DataFrame, values: pd.Series) -> None:
        self.data = data                    # Dataframe for data, one column per feature
        self.values = values                # Series for the avtual values
        self.features = self.data.columns   # List of all features in the dataset
        self.X = self.data.copy(deep=True)  # Copy the data to a new dataframe
        self.X.insert(0, 'const', 1)        # Initialise the constnt component of the matrix
        self.B = []                         # The betas (coefficients) we will find
        self.bIterations = []               # The iterations of the betas for gradient descent
    
    def dataToBinary(self) -> any:
        """
        Description:
        Takes two binary string labelled data into binary digits
        
        Parameters:
        None
        """
        
        print("Transforming data to binary")
        # Can potentially make this global, or add to logisticRegression.py
        for index, col in enumerate(self.features):
            uniqueX = list(self.X.groupby(col).groups.keys())
            # if two values and not binary...
            if len(uniqueX) == 2 and len(set(uniqueX).intersection({0, 1})) != 2:
                print(f"Updating '{col}' to be binary")
                self.X.loc[self.X[col] == uniqueX[0], col] = 0; print(f"'{uniqueX[0]}' is now 0"); 
                self.X.loc[self.X[col] == uniqueX[1], col] = 1; print(f"'{uniqueX[1]}' is now 1")
        
        # Updating Y values to be binary
        uniqueY = list(self.values.unique())
        if len(uniqueY) == 2 and len(set(uniqueY).intersection({0, 1})) != 2:
            print(f"Updating Values to be binary")
            self.values.loc[self.values == uniqueY[0]] = 0; print(f"{uniqueY[0]} is now 0")
            self.values.loc[self.values == uniqueY[1]] = 1; print(f"{uniqueY[1]} is now 1")
            
        return self
    
    def scaleData(self) -> None:
        """
        Description:
        Scales the data along by the mean and squashes in by the variance.
        This essentially normalises the data so when the nearest neighbour
        algo is run, the difference values are weighted correctly.
        
        Parameters:
        None
        """
        print("Scaling data...")
        self.scaledData, self.mean, self.variance = scaleData(self.data)
        self.X = self.scaledData.copy(deep=True)
        self.X.insert(0, 'const', 1)
    
    def _overHalf(val: float) -> int:
        """
        Description:
        'Private' function for use after the sigmoid to classify
        
        Parameters:
        val (float): number to evaluate
        """
        return 1 if val > 0.5 else 1
    
    def maxiumumLikelihood(self, alpha: float, threshold: float, useScaled=False) -> any:
        """
        Description:
        Apply the gradient descent algorithm to find the hyperplane that best intersects the data.
        This will be different to the gradient descent in LinearRegression, we will approach it
        by MAXimizing the log likelihood partial differential equations for the
        
        Parameters:
        alpha (float): Learning rate of the function
        threshold (float): error interval to run the function until
        """
        # Scale data if kwrarg flag set
        if useScaled:
            # potentially remove binary
            self.dataToBinary()
            self.scaleData()
            
        
        # Initialise the betas, between -1 and 1
        self.B = pd.Series(np.random.random(len(self.X.columns)))
        print(self.B)
        
        # Run the iterative update to the betas...
        startLoopCount = 20
        for i in range(startLoopCount):
            # for each beta / feature
            for j in range(len(self.B)):
                
                # Difference between actual and sigmoid
                valDiff = np.zeros(len(self.values))
                for val in range(len(self.values)):
                    valDiff[val] = self.values.iloc[val] - sigmoid(self.X.iloc[val, :].to_numpy(), self.B.to_numpy())
                
                # Update betas
                self.B[j] = self.B[j] + alpha * (1 / self.X.shape[0]) * np.multiply(valDiff, self.X.iloc[:, j].to_numpy()).sum()
            # print(self.B)
            
        return self
        
    def showPlot(self) -> any:
        """
        """
        assert len(self.B) > 0, "Please run gradient descent to generate betas for the hyperplane"
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of the first two columns and the regression line
        cols = self.X.columns
        scatter = ax.scatter(
            self.X[cols[1]],
            self.X[cols[2]],
            self.X[cols[3]],
            c=self.values,
            marker='o'
        )
        
        # Set labels
        ax.set_title("Logistic Regression Model")
        ax.set_xlabel(cols[1]); ax.set_ylabel(cols[2]); ax.set_zlabel(cols[3])
        # Animation function to update view angle
        def update(frame):
            ax.view_init(elev=30, azim=frame)  # Change azimuth angle
            return ax,

        # Create animation
        # self.ani =  FuncAnimation(fig, update, frames=360, interval=30, blit=False)
        
        plt.show()        
        return self
