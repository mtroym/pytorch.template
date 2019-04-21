import numpy as np

def IoU(Reframe, GTframe):
    """
    Reframe: result (Xmid, Ymid, height, width)
    GTframe: ground truth (Xmid, Ymid, height, width)
    """
    result = 0
    batchSize = GTframe.shape[0]
    result = np.zeros(batchSize)

    for i in range(batchSize):
        x1, y1, width1, height1 = Reframe[i]
        x2, y2, width2, height2 = GTframe[i]
        x1 *= 224
        x2 *= 224
        y1 *= 224
        y2 *= 224
        width1 = np.exp(width1) * 224
        width2 = np.exp(width2) * 224
        height1 = np.exp(height1) * 224
        height2 = np.exp(height2) * 224

        endx = max(x1 + width1 / 2.0, x2 + width2 / 2.0)
        startx = min(x1 - width1 / 2.0, x2 - width2 / 2.0)
        width = width1 + width2 - ( endx - startx )

        endy = max(y1 + height1 / 2, y2 + height2 / 2)
        starty = min(y1 - height1 / 2, y2 - height2 / 2)
        height = height1 + height2 - ( endy - starty )

        if width <= 0 or height <= 0:
            ratio = 0
        else:
            Area = width * height
            Area1 = width1 * height1
            Area2 = width2 * height2
            ratio = Area * 1. / (Area1+Area2-Area)
        result[i] = ratio
    return result


def ValIoU(Reframe, GTframe):
    """
    Reframe: result (Xmid, Ymid, height, width)
    GTframe: ground truth (b, 50, Xmid, Ymid, height, width)
    """
    result = 0
    batchSize = GTframe.shape[0]
    result = np.zeros(batchSize)
    
    targets = np.zeros((batchSize, 4))
    
    for i in range(batchSize):
        x1, y1, width1, height1 = Reframe[i]
        tmpIoU = 0
        for j in range(50):
            x2, y2, width2, height2 = GTframe[i, j, :]
            if abs(x2) + abs(y2) + abs(width2) + abs(height2) == 0:
#                 print(GTframe[i, j, :], j)
                break
#             print(j)
            x1 *= 224
            x2 *= 224
            y1 *= 224
            y2 *= 224
            width1 = np.exp(width1) * 224
            width2 = np.exp(width2) * 224
            height1 = np.exp(height1) * 224
            height2 = np.exp(height2) * 224

            endx = max(x1 + width1 / 2.0, x2 + width2 / 2.0)
            startx = min(x1 - width1 / 2.0, x2 - width2 / 2.0)
            width = width1 + width2 - ( endx - startx )

            endy = max(y1 + height1 / 2, y2 + height2 / 2)
            starty = min(y1 - height1 / 2, y2 - height2 / 2)
            height = height1 + height2 - ( endy - starty )


            if width <= 0 or height <= 0:
#                 print(width, height)
                tmpIoU = 0
            else:
                Area = width * height
                Area1 = width1 * height1
                Area2 = width2 * height2
                if Area * 1. / (Area1+Area2-Area) > tmpIoU:
                    tmpIoU = Area * 1. / (Area1+Area2-Area)
                    targets[i] = GTframe[i, j, :]
        result[i] = tmpIoU
#     print(targets)
    return result, targets


def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))