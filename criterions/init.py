import importlib
import os

import torch


def setup(opt, checkpoint, model):
    criterionHandler = importlib.import_module('models.' + opt.netType + '.' + opt.netType + '-criterion')
    if checkpoint != None:
        criterionPath = os.path.join(opt.resume, checkpoint['criterionFile'])
        assert os.path.exists(criterionPath), '=> WARNING: Saved criterion not found: ' + criterionPath
        print('=> Resuming criterion from ' + criterionPath)
        criterion = torch.load(criterionPath)
        criterionHandler.initCriterion(criterion, model)
        metrics = criterionHandler.createMetrics(opt, model)
    else:
        print('=> Creating criterion from file: models/' + opt.netType + '/' + opt.netType + '-criterion.py')
        criterion = criterionHandler.createCriterion(opt, model)
        metrics = criterionHandler.createMetrics(opt, model)
    if opt.GPU:
        criterion = criterion.cuda()
        try:
            metrics = metrics.cuda()
        except:
            pass
    return criterion, metrics
