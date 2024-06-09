import Datasets
from Network import *

trainingSet, validationSet = Datasets.getTrainingDatasets()

learningRates = [0.1, 0.01, 0.001]
weightDecay = [0.001, 0.0001, 0.0004]

baseParameters = Network().state_dict()

for lr in learningRates :
    for wd in weightDecay :
        graphicFilePath = ("runs/SGD_lr" + str(lr) + " _wd" + str(wd) + ".pdf")
        textFilePath = ("runs/SGD_lr" + str(lr) + " _wd" + str(wd) + ".txt")
        title = ("SGDM - Learning rate: " + str(lr) + " - Weight Decay: " + str(wd))
        sgdModel = Network()
        sgdModel.load_state_dict(baseParameters)
        trainModel(sgdModel, trainingSet, validationSet, 
            learningRate=lr, weightDecay=wd, optimizer="SGDM", epochs=10)
        opt = torch.optim.SGD(sgdModel.parameters(), lr=lr, weight_decay=wd)
        plotStatistics(sgdModel, title=title, path=graphicFilePath)
        printStatistics(sgdModel, textFilePath)
        
    
    

