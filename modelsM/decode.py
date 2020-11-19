from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import sys

# from .utils import _gather_feat, _tranpose_and_gather_feat
# import sys
# sys.path.append("/home/zt/workspace/FairMOT/src/lib/models/networks/")
# from utils import _gather_feat, _tranpose_and_gather_feat

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    # print('in nms1 ', heat[0,0,:10, :10])
    # # hmax = nn.functional.max_pool2d(
    # #     heat, (kernel, kernel), stride=1, padding=pad)
    # height = heat.size(2)/4
    # matix0 = heat[0,0,]
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    # print('in nms2 ', (heat* keep)[0,0,:10, :10])
    return heat * keep

def _nms1(heat, kernel=3):
    pad = (kernel - 1) // 2

    # print('in nms1 ', heat[0,0,:10, :10])
    # # hmax = nn.functional.max_pool2d(
    # #     heat, (kernel, kernel), stride=1, padding=pad)
    height = int(heat.size()[2]/2)
    print(height)
    matix0 = heat[:,:,:height, :]
    matix1 = heat[:,:,height:height*2, :]
    print(matix0.shape, matix1.shape)
    hmax1 = nn.functional.max_pool2d(matix0, (1, 3), stride=1, padding=(0,1))
    # print(hmax1.shape)
    hmax2 = nn.functional.max_pool2d(matix1, (kernel, kernel), stride=1, padding=pad)
    print(hmax2.shape)
    # hmax = torch.cat((matix0, hmax1), 2)
    hmax = torch.cat((hmax1, hmax2), 2)
    # hmax = nn.functional.max_pool2d(
    #     heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    # print('in nms2 ', (heat* keep)[0,0,:10, :10])
    return heat * keep


def _topk_channel(scores, K=40):
      batch, cat, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds / width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def mot_decode(heat, wh, reg=None, ltrb=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    # heat = _nms(heat)
    heat = _nms(heat)
    # print('K', K)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if ltrb:
        wh = wh.view(batch, K, 4)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    if ltrb:
        bboxes = torch.cat([xs - wh[..., 0:1],
                            ys - wh[..., 1:2],
                            xs + wh[..., 2:3],
                            ys + wh[..., 3:4]], dim=2)
    else:
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections, inds
