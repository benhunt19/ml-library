# This file is for the main execution of the code

# Import all classes and tests
from core.routers.classRouter import *
from core.routers.testRouter import *

def main() -> None:
    
    # GRADIENT DESCENT
    # testGradientDescent()
    # testAnalyticalSolution()
    # testPlotting()
    # testPlottingPlotly()

    # K NEAREST NEIGHBOURS
    # testScaleData()
    # testKNearestNeighbours()
    # testHyperParameterRegression()
    # testVisuliseNeighboursPlotly()
    
    # LOGISTIC REGRESSION
    # testLogisticRegressionBinaryUpdate()
    # testLogisticRegressionMaxiumumLikelihood()
    # testLogisticRegressionPlot()
    # plotlyTestAnimate()
    
    # AUTO REGRESSION
    # testSimulateAR1()
    # testSimulateARN()
    # testSimulateVAR()
    # testSimulateVAR3d()
    testSimulateVARrand()

# Main executable
if __name__ == '__main__':
    main()