import tensorboardX


class BoardX:
    def __init__(self, opt):
        self.writer = tensorboardX.SummaryWriter(log_dir=opt.logDir + '_' + opt.hashKey)

    def close(self):
        self.writer.close()