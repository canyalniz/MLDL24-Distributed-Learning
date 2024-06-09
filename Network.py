import sys
import torch
import torch.nn
import torch.optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as pyplot

# Model class

class Network(torch.nn.Module) :
    def __init__(self) :
        super().__init__()
        self.network_stack = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 5),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, 5),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(1600, 384),
            torch.nn.Linear(384, 192),
            torch.nn.Linear(192, 100)
        )
        self.trainingLoss = list()
        self.trainingAccuracy = list()
        self.validationLoss = list()
        self.validationAccuracy = list()
        self.to("cuda")
        
    def forward(self, input) :
        return self.network_stack(input)


# Function for training the model

def trainModel(model, trainingSet, validationSet, learningRate, weightDecay,
            optimizer, learningRateScheduler=None, batchSize=64, epochs=150) :
    trainLoader = DataLoader(trainingSet, batch_size=batchSize)
    validationLoader = DataLoader(validationSet, batch_size=batchSize)
    if optimizer == "SGDM" :
        optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, weight_decay=weightDecay)
    elif optimizer == "AdamW" :
        optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate, weight_decay=weightDecay)
    if learningRateScheduler == None :
        learningRateScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    lossFunction = torch.nn.CrossEntropyLoss()
    for ep in range(epochs) :
        totalLossValue = 0
        successfulPredictions = 0
        model.train()
        for data, labels in trainLoader :
            data = data.to("cuda")
            labels = labels.to("cuda")
            optimizer.zero_grad()
            output = model(data)
            lossValue = lossFunction(output, labels)
            lossValue.backward()
            optimizer.step()
            totalLossValue += lossValue.item()
            predictions = torch.argmax(output, 1)
            for i in range(labels.size(0)) :
                if labels[i] == predictions[i] :
                    successfulPredictions += 1
        model.trainingLoss.append(totalLossValue)
        model.trainingAccuracy.append(successfulPredictions / len(trainingSet))
        learningRateScheduler.step()
        totalLossValue = 0
        successfulPredictions = 0
        model.eval()
        for data, labels in validationLoader :
            data = data.to(device="cuda")
            labels = labels.to("cuda")
            output = model(data)
            lossValue = lossFunction(output, labels)
            totalLossValue += lossValue.item()
            predictions = torch.argmax(output, 1)
            for i in range(labels.size(0)) :
                if labels[i] == predictions[i] :
                    successfulPredictions += 1
        model.validationLoss.append(totalLossValue)
        model.validationAccuracy.append(successfulPredictions / len(validationSet))

# Plot statistic in pdf using matplotlib

def plotStatistics(model, title, path) :
    figure = pyplot.figure(figsize=(10,10))
    figure.suptitle(title)
    lossPlot = figure.add_subplot(2, 1, 1)
    lossPlot.set_title("Losses")
    lossPlot.plot(range(len(model.trainingLoss)), model.trainingLoss, label="Training")
    lossPlot.plot(range(len(model.validationLoss)), model.validationLoss, label="Validation")
    lossPlot.legend()
    accuracyPlot = figure.add_subplot(2, 1, 2)
    accuracyPlot.set_title("Accuracies")
    accuracyPlot.plot(range(len(model.trainingAccuracy)), model.trainingAccuracy, label="Training")
    accuracyPlot.plot(range(len(model.validationAccuracy)), model.validationAccuracy, label="Validation")
    accuracyPlot.legend()
    figure.savefig(path) 


# Print statistic in a text file 

def printStatistics(model, filePath) :
    outputFile = open(filePath, "w")
    outputFile.write("Training loss\n")
    outputFile.write(str(model.trainingLoss))
    outputFile.write("\nValidation loss\n")
    outputFile.write(str(model.validationLoss))
    outputFile.write("\nTraining accuracy\n")
    outputFile.write(str(model.trainingAccuracy))
    outputFile.write("\nValidation accuracy\n")
    outputFile.write(str(model.validationAccuracy))
    outputFile.close()

