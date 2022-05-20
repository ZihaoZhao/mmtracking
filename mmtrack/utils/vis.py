#----------------description----------------# 
# Author       : Zihao Zhao
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : IBICAS, Fudan University
# Date         : 2022-05-20 11:39:00
# LastEditors  : Zihao Zhao
# LastEditTime : 2022-05-20 14:38:19
# FilePath     : /mmtracking/mmtrack/utils/vis.py
# Description  : 
#-------------------------------------------# 

import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
import cv2

def save_heatmap(filename, nparray2d, bbox_list=[], bbox_format="xyxy"):
    """
    bbox_list  [(1,4)]
    """


    #"/zhzhao/code/mmtracking_master_20220513/sys_log/kernel.png"
    print("save_heatmmap:", filename)
    print("draw", bbox_list)
    if bbox_format == "xywh":
        kernel_vis = copy.deepcopy(nparray2d)
        for bi, bbox in enumerate(bbox_list):
            kernel_vis=cv2.rectangle(kernel_vis, (int(bbox[0][0]), int(bbox[0][1])), \
                                                (int(bbox[0][0]+bbox[0][2]), int(bbox[0][1]+bbox[0][3])), (0), 2)

    elif bbox_format == "cxywh":
        kernel_vis = copy.deepcopy(nparray2d)
        for bi, bbox in enumerate(bbox_list):
            kernel_vis=cv2.rectangle(kernel_vis, (int(bbox[0][0]-bbox[0][2]//2), int(bbox[0][1]-bbox[0][3]//2)), \
                                                (int(bbox[0][0]+bbox[0][2]//2), int(bbox[0][1]+bbox[0][3]//2)), (0), 2)

    elif bbox_format == "xyxy":
        kernel_vis = copy.deepcopy(nparray2d)
        for bi, bbox in enumerate(bbox_list):
            kernel_vis=cv2.rectangle(kernel_vis, (int(bbox[0][0]), int(bbox[0][1])), \
                                                (int(bbox[0][2]), int(bbox[0][3])), (0), 2)

    elif bbox_format == "0xyxy":
        kernel_vis = copy.deepcopy(nparray2d)
        for bi, bbox in enumerate(bbox_list):
            kernel_vis=cv2.rectangle(kernel_vis, (int(bbox[0][1]), int(bbox[0][2])), \
                                                (int(bbox[0][3]), int(bbox[0][4])), (0), 2)

    sns.heatmap(kernel_vis)
    plt.savefig(filename) 
    plt.close()
