# -*- coding: utf-8 -*-
import numpy as np
import os
import SimpleITK as sitk
import nibabel as nib


def file_name(file_dir):
    L = []
    path_list = os.listdir(file_dir)
    path_list.sort()  # 对读取的路径进行排序
    for filename in path_list:
        if filename.endswith('.nii.gz'):
            L.append(os.path.join(filename))
    return L


def computeQualityMeasures(lP, lT):
    quality = dict()
    labelPred = sitk.GetArrayFromImage(lP)
    labelTrue = sitk.GetArrayFromImage(lT)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()

    P = sitk.GetImageFromArray((labelTrue == 1).astype(np.int))
    GT = sitk.GetImageFromArray((labelPred == 1).astype(np.int))

    hausdorffcomputer.Execute(P, GT)
    #  More less more ok.
    quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()

    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(P, GT)
    quality["dice"] = dicecomputer.GetDiceCoefficient()
    return quality




if __name__ == '__main__':

    GT_PTH = '/Users/tony/Downloads/train/Patient_40/'
    gt_names = file_name(GT_PTH)[:1]
    print(gt_names)
    pd_names = gt_names
    NUM = []
    P = []
    for i in range(len(gt_names)):
        print(GT_PTH + gt_names[i])
        gt = sitk.ReadImage(GT_PTH + gt_names[i])
        pred = sitk.ReadImage(GT_PTH + gt_names[i])
        print('Load file - ok')
        quality = computeQualityMeasures(gt, pred)
        print(quality)