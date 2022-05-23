#----------------description----------------# 
# Author       : Zihao Zhao
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : IBICAS, Fudan University
# Date         : 2022-05-14 14:51:50
# LastEditors  : Zihao Zhao
# LastEditTime : 2022-05-23 22:03:05
# FilePath     : /mmtracking/mmtrack/datasets/pipelines/__init__.py
# Description  : 
#-------------------------------------------# 
# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.builder import PIPELINES

from .formatting import (CheckPadMaskValidity, ConcatSameTypeFrames,
                         ConcatVideoReferences, ReIDFormatBundle,
                         SeqDefaultFormatBundle, ToList, VideoCollect)
from .loading import (LoadDetections, LoadMultiImagesFromFile,
                      SeqLoadAnnotations, MSLoadImageFromFile, MSLoadAnnotations)
from .processing import MatchInstances, PairSampling, TridentSampling
from .transforms import (SeqBboxJitter, SeqBlurAug, SeqBrightnessAug,
                         SeqColorAug, SeqCropLikeSiamFC, SeqCropLikeSiamFC, SeqCropLikeStark,
                         SeqGrayAug, SeqNormalize, SeqPad,
                         SeqPhotoMetricDistortion, SeqRandomCrop,
                         SeqRandomFlip, SeqResize, SeqShiftScaleAug, MSSeqShiftScaleAug)

__all__ = [
    'PIPELINES', 'LoadMultiImagesFromFile', 'SeqLoadAnnotations', 'SeqResize', 'MSLoadImageFromFile', 'MSLoadAnnotations'
    'SeqNormalize', 'SeqRandomFlip', 'SeqPad', 'SeqDefaultFormatBundle',
    'VideoCollect', 'CheckPadMaskValidity', 'ConcatVideoReferences',
    'LoadDetections', 'MatchInstances', 'SeqRandomCrop',
    'SeqPhotoMetricDistortion', 'SeqCropLikeSiamFC', 'SeqCropLikeSiamFC', 'SeqShiftScaleAug', 'MSSeqShiftScaleAug',
    'SeqBlurAug', 'SeqColorAug', 'ToList', 'ReIDFormatBundle', 'SeqGrayAug',
    'SeqBrightnessAug', 'SeqBboxJitter', 'SeqCropLikeStark', 'TridentSampling',
    'ConcatSameTypeFrames', 'PairSampling'
]
