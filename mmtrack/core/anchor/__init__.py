#----------------description----------------# 
# Author       : Zihao Zhao
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : IBICAS, Fudan University
# Date         : 2022-05-14 14:51:50
# LastEditors  : Zihao Zhao
# LastEditTime : 2022-05-16 12:55:38
# FilePath     : /mmtracking/mmtrack/core/anchor/__init__.py
# Description  : 
#-------------------------------------------# 
# Copyright (c) OpenMMLab. All rights reserved.
from .sot_anchor_generator import SiameseRPNAnchorGenerator, MSTrackerAnchorGenerator

__all__ = ['SiameseRPNAnchorGenerator', 'MSTrackerAnchorGenerator']
