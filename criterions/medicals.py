# -*- coding: utf-8 -*-
import numpy as np
import os
import SimpleITK as sitk


def file_name(file_dir):
    L = []
    path_list = os.listdir(file_dir)
    path_list.sort()  # 对读取的路径进行排序
    for filename in path_list:
        if 'nii' in filename:
            L.append(os.path.join(filename))
    return L


def computeQualityMeasures(lP, lT):
    quality = dict()
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()

    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    quality["dice"] = dicecomputer.GetDiceCoefficient()

    return quality




if __name__ == '__main__':

    # gtpath = ''
    # predpath = ''
    #
    # gtnames = file_name(gtpath)
    # prednames = file_name(predpath)
    #
    # labels_num = np.zeros(len(prednames))
    # NUM = []
    # P = []
    #
    # for i in range(len(gtnames)):
    #     gt = sitk.ReadImage(gtpath + gtnames[i])
    #     pred = sitk.ReadImage(predpath + gtnames[i])
    #     quality = computeQualityMeasures(pred, gt)


    gtpath = ''


