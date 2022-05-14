#----------------description----------------# 
# Author       : Zihao Zhao
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : IBICAS, Fudan University
# Date         : 2022-05-14 14:52:25
# LastEditors  : Zihao Zhao
# LastEditTime : 2022-05-14 14:59:36
# FilePath     : /mmtracking/mmtrack/models/track_heads/__init__.py
# Description  : 
#-------------------------------------------# 
# Copyright (c) OpenMMLab. All rights reserved.
from .quasi_dense_embed_head import QuasiDenseEmbedHead
from .quasi_dense_track_head import QuasiDenseTrackHead
from .roi_embed_head import RoIEmbedHead
from .roi_track_head import RoITrackHead
from .siamese_rpn_head import CorrelationHead, SiameseRPNHead
from .stark_head import CornerPredictorHead, StarkHead
from .mstracker_head import MCorrelationHead, MSTrackerHead

__all__ = [
    'CorrelationHead', 'MCorrelationHead', 'SiameseRPNHead', 'MSTrackerHead', 'RoIEmbedHead', 'RoITrackHead',
    'StarkHead', 'CornerPredictorHead', 'QuasiDenseEmbedHead',
    'QuasiDenseTrackHead'
]
