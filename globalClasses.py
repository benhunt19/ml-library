# This will be where we store global functions to use within the library

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LinearRegression:
    def __init__(self, data):
        self.test = "Hello World"     # Delete once into production
        self.data = data              # Data should be a pandas dataframe
        self.X = self.data.copy(deep=True)      # Copy the data to a new dataframe
        self.X.insert(0, 'const', 1)
        print(self.X)
    
    def gradientDescent(self, alpha, iterations):
        # Get the columns of the data
        features = self.X.columns[:len(self.X.columns)-1]
        y = self.X.columns[len(self.data.columns)-1] # the 'labels' of the previous columns
        N = self.data.shape[0]
        B = np.array([0.0 for i in range(len(features))]) # Initialize the betas
        # For each iteration
        for i in range(iterations):
            # for each liner value in B
            for b in range(len(B)):
                # run process
                tmpDelta = 0
                for n in range(N):
                    tmpMiniSum = 0
                    for j in range(len(features)):
                        tmpMiniSum += B[j] * self.X[features[j]][n]
                    tmpDelta += (tmpMiniSum - self.X[y][n]) * self.X[features[b]][n]
                B[b] = B[b] - alpha * (1/N) * tmpDelta
        print(B) 
                     

    def showPlot(self):
        # Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of the first two columns and the regression line
        cols = self.data.columns
        ax.scatter(self.data[cols[0]],self.data[cols[1]], self.data[cols[2]], c='r', marker='o')
        # Set labels
        ax.set_title("Liniear Regression Test")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

        plt.show()