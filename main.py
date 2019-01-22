import sys
import opts
import math
import importlib
print("=> initializing. parsing arguments.")
opt = opts.parse()

####################################################
models = importlib.import_module('models.init')
criterions = importlib.import_module('criterions.init')
checkpoints = importlib.import_module('checkpoints')
Trainer = importlib.import_module('models.' + opt.netType + '.' + opt.netType + '-train')
try:
    # print("\t=> trying import from network repos.")
    DataLoader = importlib.import_module('models.' + opt.netType + '.' + opt.netType + '-dataloader')
except ImportError:
    # print("\t=>not found in network repos. try to import from datasets repos.")
    DataLoader = importlib.import_module('datasets.dataloader')
####################################################

# Data loading
print('=> Setting up data loader')
trainLoader, valLoader = DataLoader.create(opt)



# Load previous checkpoint, if it exists
print('=> Checking checkpoints')
checkpoint = checkpoints.load(opt)

# Create model
model, optimState = models.setup(opt, checkpoint)
criterion, metric = criterions.setup(opt, checkpoint, model)
# The trainer handles the training loop and evaluation on validation set
trainer = Trainer.createTrainer(model, criterion, metric, opt, optimState)

if opt.testOnly:
    loss = trainer.test(valLoader, 0)
    sys.exit()

bestLoss = math.inf
startEpoch = max([1, opt.epochNum])
if checkpoint != None:
    startEpoch = checkpoint['epoch'] + 1
    bestLoss = checkpoint['loss']
    print('Previous loss: \033[1;36m%1.4f\033[0m' % bestLoss)

trainer.LRDecay(startEpoch)

for epoch in range(startEpoch, opt.nEpochs + 1):
    trainer.LRDecayStep()

    trainLoss = trainer.train(trainLoader, epoch)
    testLoss = trainer.test(valLoader, epoch)

    bestModel = False
    if testLoss < bestLoss:
        bestModel = True
        bestLoss = testLoss
        print(' * Best model: \033[1;36m%1.4f\033[0m * ' % testLoss)

    checkpoints.save(epoch, trainer.model, criterion, metric, trainer.optimizer, bestModel, testLoss ,opt)

print(' * Finished Err: \033[1;36m%1.4f\033[0m * ' % bestLoss)
