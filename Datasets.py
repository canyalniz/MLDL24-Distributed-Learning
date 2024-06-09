import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def getTrainingDatasets(validationPercentage=0.2) :
    trainingDataset = datasets.CIFAR100(root="./Datasets",train=True,download=False,transform=ToTensor())
    randomIndexes = random.sample(range(0, len(trainingDataset)), round(len(trainingDataset) * validationPercentage))
    trainingSet = [trainingDataset[i] for i in range(len(trainingDataset)) if i not in randomIndexes]
    validationSet = [trainingDataset[i] for i in randomIndexes]
    return trainingSet, validationSet

def getTestSet() :
    return datasets.CIFAR100(root="./Datasets",train=False,download=False,transform=ToTensor())



