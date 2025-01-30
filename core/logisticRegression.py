import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from core.globalFunctions import sigmoid

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
    
    def _overHalf(val: float) -> int:
        """
        Description:
        'Private' function for use after the sigmoid to classify
        
        Parameters:
        val (float): number to evaluate
        """
        return 1 if val > 0.5 else 1
    
    def gradientDescent(self, alpha: float, threshold: float) -> any:
        """
        Description:
        Apply the gradient descent algorithm to find the hyperplane that best intersects the data.
        This will be different to the gradient descent in LinearRegression, we will approach it
        by MAXimizing the log likelihood partial differential equations for the
        
        Parameters:
        alpha (float): Learning rate of the function
        threshold (float): error interval to run the function until
        """
        
        # Initialise the betas, between -1 and 1
        self.B = pd.Series(np.random.random(len(self.X.columns)) * 2 - 1)
        print(self.B)
        
        # Run the iterative update to the betas...
        startLooCount = 100
        for i in range(startLooCount):
            # for each beta / feature
            for j in range(len(self.B)):
                
                # Difference between actual and sigmoid
                valDiff = np.zeros(len(self.values))
                for val in range(len(self.values)):
                    valDiff[val] = self.values.iloc[val] - sigmoid(self.X.iloc[val, :].to_numpy(), self.B.to_numpy())
                
                # Update betas
                np.multiply(valDiff, self.X.iloc[:, j].to_numpy()).sum()
                
        print(self.B)
        return self
        
    def showPlot(self) -> any:
        
        # Define the normal vector (a, b, c)
        normal_vector = np.array([1, 2, 3])  # Example normal vector

        # Define a point on the plane (x0, y0, z0)
        point_on_plane = np.array([1, 1, 1])  # Example point on the plane

        # Extract components of the normal vector
        a, b, c = normal_vector

        # Extract the point (x0, y0, z0)
        x0, y0, z0 = point_on_plane

        # Create a mesh grid for x and y values
        x = np.linspace(-10, 10, 400)
        y = np.linspace(-10, 10, 400)
        X, Y = np.meshgrid(x, y)

        # Calculate the corresponding Z values using the plane equation
        Z = -(a * (X - x0) + b * (Y - y0)) / c + z0

        # Plotting the surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap='viridis')

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Surface with Normal Vector')

        plt.show()
        
        return self
