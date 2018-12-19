import torch.nn as nn

def initCriterion(criterion, model):
    pass

def createCriterion(opt, model):
    criterion = nn.MSELoss()

    return criterion
