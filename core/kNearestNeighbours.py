import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

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
        self.values = values                # Series for the avtual values
        