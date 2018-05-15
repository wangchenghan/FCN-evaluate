#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 00:00:54 2018

@author: wang

用于FCN结果评估的代码，包含pixel accuarcy,mean accuarcy,mean IU,frequency weighted IU指标的评估
以及用于整体评估的代码。

"""
from PIL import Image
import numpy as np
import os

def get_pic(image):
    """
    将输入的str或是Image对象统一为Image对象

    parameter
    ---------
    image : str or Image object
         输入图像
    
    return
    ---------
    out : Image object
         Image对象
    """
    if isinstance(image, str):
        return Image.open(image)
    return image

def get_classes(pic):
    """
    获取pic中颜色的数量，或者说灰度图像中所代表的的类别
    """
    classes = []
    color_list = pic.getcolors()
    for color in range(len(color_list)):
        classes.append(color_list[color][1])
    return classes


def Pii(gt_array, pd_array, classes):
    """
    找classes类中的Pii，即判断正确的结果，将其总数返回
    """
    m = np.size(gt_array,axis = 0)
    n = np.size(gt_array,axis = 1)
    
    sum = 0
    for i in range(m):
        for j in range(n):
            if gt_array[i][j] == classes and pd_array[i][j] == classes:
                sum += 1
    return sum


              
def get_union(array1, array2, classes):
    """
    遍历两个数组，求两个数组中具有classes值的元素的数量，重合的部分只统计一次。

    parameter
    ---------
    array1 : np.array object
      数组1
    array2 : np.array object
      数组2
    classes: int
    
    return
    ---------
    out : int
      数量
    """
    m = np.size(array1, axis=0)
    n = np.size(array1, axis=1)
    
    sum = 0
    for i in range(m):
        for j in range(n):
            if array1[i][j] == classes or array2[i][j] == classes:
                sum += 1
    return sum


def pixel_accuarcy(ground_truth, prediction):
    """
    求得ground_truth和prediction的pixel_accuarcy

    parameter
    ---------
    ground_truth : str or Image object
        标注图像,且应为灰度图像
        prediction ：str or Image object
        预测图像，且应为灰度图像
    
    return
    ---------
    pa : float or None
         pixel_accuarcy值
    """
    #输入图像
    gt = get_pic(ground_truth)
    pd = get_pic(prediction)
    
    #获取ground_truth类别
    classes = get_classes(gt)

    #将图像转为数组
    gt_a = np.array(gt)
    pd_a = np.array(pd)
    
    #求Pii
    sum_Pii = 0
    for i in range(len(classes)):
        sum_Pii += Pii(gt_a, pd_a, classes[i])
    
    #按公式求PA
    PA = sum_Pii/(gt.size[0]*gt.size[1])
    return PA

def mean_pixel_accuracy(ground_truth, prediction):
    """
    求得ground_truth和prediction的mean_pixel_accuarcy

    parameter
    ---------
    ground_truth : str or Image object
        标注图像,且应为灰度图像
        prediction ：str or Image object
        预测图像，且应为灰度图像
    
    return
    ---------
    pa : float
         mean_pixel_accuarcy值
    """
    #输入图像
    gt = get_pic(ground_truth)
    pd = get_pic(prediction)
    
    #获取ground_truth类别
    classes = get_classes(gt)

    #将图像转为数组
    gt_a = np.array(gt)
    pd_a = np.array(pd)

    #按公式求PA
    sum = 0
    for i in range(len(classes)):
        single_Pii = Pii(gt_a, pd_a, classes[i])
        sum += single_Pii/gt.getcolors()[i][0]
    MPA = sum/len(classes)    
    return MPA

    
def mean_IU(ground_truth, prediction):
    """
    求得ground_truth和prediction的pixel_accuarcy

    parameter
    ---------
    ground_truth : str or Image object
        标注图像,且应为灰度图像
        prediction ：str or Image object
        预测图像，且应为灰度图像
    
    return
    ---------
    pa : float
         pixel_accuarcy值
    """
    #输入图像
    gt = get_pic(ground_truth)
    pd = get_pic(prediction)
    
    #获取ground_truth类别
    classes = get_classes(gt)
    
    #将图像转为数组
    gt_a = np.array(gt)
    pd_a = np.array(pd)
    
    #按公式计算MIoU
    sum = 0
    for i in range(len(classes)):
        single_Pii = Pii(gt_a, pd_a, classes[i])
        sum += single_Pii/get_union(gt_a, pd_a, classes[i])
    MIoU = sum/len(classes)
    return MIoU

def frequency_weighted_IU(ground_truth, prediction):
    """
    求得ground_truth和prediction的Frequency Weighted Intersection over Union

    parameter
    ---------
    ground_truth : str or Image object
        标注图像,且应为灰度图像
        prediction ：str or Image object
        预测图像，且应为灰度图像
    
    return
    ---------
    pa : float
        FWIoU值
    """
    #输入图像
    gt = get_pic(ground_truth)
    pd = get_pic(prediction)
    
    #获取ground_truth类别
    classes = get_classes(gt)
    
    #将图像转为数组
    gt_a = np.array(gt)
    pd_a = np.array(pd)
    
    #按公式计算MIoU
    sum = 0
    for i in range(len(classes)):
        single_Pii = Pii(gt_a, pd_a, classes[i])
        sum += (gt.getcolors()[i][0]*single_Pii)/get_union(gt_a, pd_a, classes[i])
    FWIoU = sum/(gt.size[0]*gt.size[1])
    return FWIoU
#def test():
#   dir1 = 'D:/9527/2018.4.12 语义分割评估（未完成）/DJI_0605.png'
#   dir2 = 'D:/9527/2018.4.12 语义分割评估（未完成）/pred_9.png'
#   dir3 = 'D:/FCN.tensorflow-master-123/test2018.4.26/A/gt/gt_5.png'
#   dir4 = 'D:/FCN.tensorflow-master-123/test2018.4.26/A/pred/pred_5.png'
#   print('PA = ',pixel_accuarcy(dir1, dir2))
#   print('MPA = ',mean_pixel_accuracy(dir3, dir4))
#   print('MIoU = ',mean_IU(dir3, dir4))
#   print('FWIoU = ',frequency_weighted_IU(dir1, dir2))

def create_image_lists(image_dir):
    image_list = {}
    list_gt = os.listdir(image_dir + '/' + 'gt')
    list_pred = os.listdir(image_dir + '/' + 'pred')
    image_list = {'gt':list_gt, 'pred':list_pred}
    
    return image_list

def FCN_evaluate(dir1):
    """
    四项评估
    
    parameter
    ---------
    dir : str
    文件夹地址，要求内含名为gt和pred两个文件夹，其中一个放标注，一个放预测
    
    return
    ---------
    result : dict
    包含dir1最后一个地址作为文件夹名，以及四个评估指标的结果，其格式为
    result = {'name': name, 'PA':PA, 'MPA':MPA, 'MIU':MIU, 'FWIoU':FWIoU}
    """
    PA = 0
    MPA = 0
    MIU = 0
    FWIoU = 0
    print('the directory is:',dir1)
    image_lists = create_image_lists(dir1)

    for i in range(len(image_lists['gt'])):
        print(i)
        ground_truth = dir1 +'/gt/'+image_lists['gt'][i]
        prediction = dir1 + '/pred/' + image_lists['pred'][i]
        PA += pixel_accuarcy(ground_truth, prediction)
        MPA += mean_pixel_accuracy(ground_truth, prediction)
        MIU += mean_IU(ground_truth, prediction)
        FWIoU += frequency_weighted_IU(ground_truth, prediction)
#        PA = pixel_accuarcy(ground_truth, prediction)
#        MPA = mean_pixel_accuracy(ground_truth, prediction)
#        MIU = mean_IU(ground_truth, prediction)
#        FWIoU = frequency_weighted_IU(ground_truth, prediction)
#        print("PA = ",PA,'\n','MPA = ',MPA, '\n', "MIU = " ,MIU,'\n','FWIoU = ',FWIoU)
    name = dir1.split('/')[-1]
    PA = PA / len(image_lists['gt'])
    MPA = MPA/len(image_lists['gt'])
    MIU = MIU /len(image_lists['gt'])
    FWIoU = FWIoU/len(image_lists['gt'])
    
    result = {'name': name, 'PA':PA, 'MPA':MPA, 'MIU':MIU, 'FWIoU':FWIoU}
    return result

if __name__ == '__main__':
    dir1 = [
#           'D:/FCN.tensorflow-master-123/test2018.4.26/A',
#           'D:/FCN.tensorflow-master-123/test2018.4.26/B1',
#           'D:/FCN.tensorflow-master-123/test2018.4.26/B2',
#           'D:/FCN.tensorflow-master-123/test2018.4.26/B3',
#           'D:/FCN.tensorflow-master-123/test2018.4.26/C',
#           'D:/FCN.tensorflow-master-123/test2018.4.26/A/Kmeans'
           'D:/FCN.tensorflow-master-123/test2018.4.26/A/HSV'
           ]

    for item in dir1:
        print(FCN_evaluate(item))
        
    
