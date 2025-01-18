# This file is for the main execution of the code

from globalClasses import *
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    
    
    # generate random datapoints
    maxInt = 50
    x = np.random.rand(100) * maxInt
    x = np.floor(x)
    variance = 30
    y = x * 2 + np.random.rand(100) * variance
    z = x + y + np.random.rand(100) * variance
    #print(x, y, z)
    
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z
    })
    print(df)
    
    linReg = LinearRegression(
        data=df
    )
    linReg.gradientDescent(0.00001, 20)