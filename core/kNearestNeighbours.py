import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from core.globalFunctions import scaleData, euclidianDistance

class KNearestNeighbours:
    """
    Description:
    Finds the distance between a data point and K of it's
    'Nearest Neighbours', depending on the classification of
    tese nearest data points, will determine the class of the
    new one
    
    Type:
    Supervised Learning
    
    Paramaters:
    data (DataFrame): the data to run regression on, there
    values (Series): the data to run regression on, there
    """
    def __init__(self, data: pd.DataFrame, values: pd.Series) -> None:
        self.data = data                    # Dataframe for data, one column per feature
        self.values = values                # Series for the actual values
        self.neighbours = pd.DataFrame([])  # Initialise a blank dataframe for the neighbours
        self.testDataPoint = None           # Create a blank test data point
    
    def scaleData(self):
        """
        Description:
        Scales the data along by the mean and squashes in by the variance.
        This essentially normalises the data so when the nearest neighbour
        algo is run, the difference values are weighted correctly.
        
        Parameters:
        None
        """
        self.scaledData, self.mean, self.variance = scaleData(self.data)
    
    def findNeighbours(self, dataPoint: pd.Series, K: int) -> pd.DataFrame:
        """
        Description:
        Find the nearest K neighbours using the euclidian distance
        
        Parameters:
        dataPoint (pd.series): find the n
        K (int): The number of integers to return
        
        """
        self.K = K
        self.testDataPoint = dataPoint.copy()
        indexStore = np.array([])
        distanceStore = np.array([])
        scaledDataPoint = dataPoint
        scaledDataPoint -= self.mean
        scaledDataPoint /= self.variance.pow(0.5) 
        print(scaledDataPoint)
        self.scaledDataPoint = scaledDataPoint
        print(dataPoint)

        for i in range(self.scaledData.shape[0]):
            distance = euclidianDistance(scaledDataPoint.squeeze(), self.scaledData.iloc[i, :])
            if len(indexStore) == 0 or len(indexStore) < K:
                indexStore = np.append(indexStore, i)
                distanceStore = np.append(distanceStore, distance)
            elif np.max(distanceStore) > distance:
                indexStore[np.argmax(distanceStore)] = i
                distanceStore[np.argmax(distanceStore)] = distance
                
        self.scaledNeighbours = self.scaledData.loc[indexStore]
                
        self.neighbours = self.data.loc[indexStore].copy()
        return self.neighbours
    
    def showPlot(self) -> None:
        """
        Description:
        Create a 3d plot of the first three dimensions
        
        Parameters:
        None
        """
        # Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of the first two columns and the regression line
        legend = ["Data"]
        ax.ticklabel_format(style='plain')
        cols = self.data.columns
        scatterAll = ax.scatter(
            self.data[cols[0]],
            self.data[cols[1]],
            self.data[cols[2]],
            c= self.data[cols[0]],
            marker='o',
            alpha=0.16  # Set transparency level
        )
        if len(self.neighbours) > 0:
            scatterNeighbours = ax.scatter(
                self.neighbours[cols[0]],
                self.neighbours[cols[1]],
                self.neighbours[cols[2]],
                c='black',
                marker='o',
                alpha=1  # Set transparency level
            )
            legend.append(f'{self.K} Nearest Neighoburs')
        
        if type(self.testDataPoint) != type(None):
            scatterTestPoint = ax.scatter(
                self.testDataPoint.loc[cols[0]],
                self.testDataPoint.loc[cols[1]],
                self.testDataPoint.loc[cols[2]],
                c='red',
                marker='o',
                alpha=1  # Set transparency level
            )
            legend.append(f'Test Data Point')
            
        ax.set_xlabel(cols[0]); ax.set_ylabel(cols[1]); ax.set_zlabel(cols[2])
        ax.legend(legend)
        ax.set_title('K Nearest Neighbours')
        # Animation function to update view angle
        def update(frame):
            ax.view_init(elev=30, azim=frame)  # Change azimuth angle
            return ax,

        # Create animation
        self.ani =  FuncAnimation(fig, update, frames=360, interval=15, blit=False)
        
        plt.show()
        
    def showPlotScaled(self) -> None:
        """
        Description:
        Create a 3d plot of the first three dimensions 
        for the scaled data
        
        Parameters:
        None
        """
        # Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of the first two columns and the regression line
        legend = ["Data"]
        ax.ticklabel_format(style='plain')
        cols = self.data.columns
        scatterAll = ax.scatter(
            self.scaledData[cols[0]],
            self.scaledData[cols[1]],
            self.scaledData[cols[2]],
            c= self.data[cols[0]],
            marker='o',
            alpha=0.16  # Set transparency level
        )
        if len(self.scaledNeighbours) > 0:
            scatterNeighbours = ax.scatter(
                self.scaledNeighbours[cols[0]],
                self.scaledNeighbours[cols[1]],
                self.scaledNeighbours[cols[2]],
                c='black',
                marker='o',
                alpha=1  # Set transparency level
            )
            legend.append(f'{self.K} Nearest Neighoburs')
        
        if type(self.testDataPoint) != type(None):
            scatterTestPoint = ax.scatter(
                self.scaledDataPoint.loc[cols[0]],
                self.scaledDataPoint.loc[cols[1]],
                self.scaledDataPoint.loc[cols[2]],
                c='red',
                marker='o',
                alpha=1  # Set transparency level
            )
            legend.append(f'Test Data Point')
            
        ax.set_xlabel(cols[0]); ax.set_ylabel(cols[1]); ax.set_zlabel(cols[2])
        ax.legend(legend)
        ax.set_title('K Nearest Neighbours')
        # Animation function to update view angle
        def update(frame):
            ax.view_init(elev=30, azim=frame)  # Change azimuth angle
            return ax,

        # Create animation
        self.ani =  FuncAnimation(fig, update, frames=120, interval=30, blit=False)
        
        plt.show()
        
    def saveAnimation(self, name: str) -> None:
        """
        Description:
        Save the plot animation as a gif
        
        Parameters:
        name (string): The name of the file
        """
        assert self.ani is not None
        self.ani.save(f'media/{name}.gif', writer=PillowWriter(fps=27))