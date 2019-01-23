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
