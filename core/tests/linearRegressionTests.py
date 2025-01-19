from core.routers.classRouter import *

def testGradientDescent() -> None:#
    """
    Description:
    Test gradient descent on a 2x featureset: x and y.
    The labels are a 1d series that are related to x and y
    
    Params:
    None
    """
    # generate random datapoints
    maxInt = 50
    c = 20 # 'y intercept'
    variance = 50
    pointCount = 200
    x = np.random.rand(pointCount) * maxInt
    y = x * 2 + np.random.rand(pointCount) * variance
    z = x + 2 * y + np.random.rand(pointCount) * variance + c
    
    data = pd.DataFrame({
        'Feature A': x,
        'Feature B': y
    })    
    values = pd.Series(z)
    linReg = LinearRegression(
        data=data,
        values=values
    )
    alpha = 0.00002; iterations = 50
    linReg.gradientDescent(alpha, iterations, linReg.X.columns)
    linReg.showPlot()

def testAnalyticalSolution() -> None:
    # generate random datapoints
    maxInt = 50
    c = 20 # 'y intercept'
    variance = 50
    pointCount = 200
    x = np.random.rand(pointCount) * maxInt
    y = x * 2 + np.random.rand(pointCount) * variance
    z = x + 2 * y + np.random.rand(pointCount) * variance + c
    
    data = pd.DataFrame({
        'Feature A': x,
        'Feature B': y
    })    
    values = pd.Series(z)
    linReg = LinearRegression(
        data=data,
        values=values
    )
    linReg.analyticalSolution()
    linReg.showPlot()

def testPlotting() -> None:
    # generate random datapoints
    maxInt = 50
    c = 20 # 'y intercept'
    variance = 50
    pointCount = 200
    x = np.random.rand(pointCount) * maxInt
    y = x * 2 + np.random.rand(pointCount) * variance
    z = x + 2 * y + np.random.rand(pointCount) * variance + c
    
    data = pd.DataFrame({
        'Feature A': x,
        'Feature B': y
    })    
    values = pd.Series(z)
    linReg = LinearRegression(
        data=data,
        values=values
    )
    linReg.showPlot()