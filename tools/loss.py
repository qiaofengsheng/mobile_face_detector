import random

import torch
from torch import nn
from tools.anchor import *

def balance_ps_ng(targets,pos_ng_scale=50):
    object_mask = targets[..., 0] > 0
    no_object_mask = targets[...,0]==0
    pos_nums = object_mask.sum()
    ng_nums = pos_nums * pos_ng_scale
    select_ng=random.sample(torch.nonzero(no_object_mask).tolist(),ng_nums)
    new_no_object_mask = torch.zeros(no_object_mask.shape, dtype=torch.bool)
    for i in select_ng:
        new_no_object_mask[i[0],i[1],i[2],i[3]]=True
    conf_mask = object_mask + new_no_object_mask
    return conf_mask

def loss_function(predicts, targets, alpha):
    '''
    :param predicts: NxHxWx14
    :param targets: NxHxWx2x6   anchor score x y w h cls
    :param alpha:
    :return:
    '''
    predicts = predicts.reshape(targets.shape[0], targets.shape[1], targets.shape[2], targets.shape[3], -1)
    object_mask = targets[..., 0] > 0
    loss_conf = nn.BCELoss().forward(torch.sigmoid(predicts[..., 0]), targets[..., 0].float())
    loss_bbox = nn.SmoothL1Loss().forward(predicts[object_mask][..., 1:5], targets[object_mask][..., 1:5])
    # loss_cls = nn.CrossEntropyLoss().forward(predicts[object_mask][..., 5:],
    #                                          targets[object_mask][..., 5:].squeeze().long())

    return loss_conf + loss_bbox


if __name__ == '__main__':
    boxes = [[1, 110, 120, 300, 400], [1, 200, 200, 500, 500]]
    c = generate_anchor_labels(boxes)
    pre1 = torch.randn((1, 80, 80, 14))
    pre2 = torch.randn((1, 40, 40, 14))
    pre3 = torch.randn((1, 20, 20, 14))
    loss_function(pre3, torch.tensor(c[0]).unsqueeze(0), '')
    loss_function(pre2, torch.tensor(c[1]).unsqueeze(0), '')
    loss_function(pre1, torch.tensor(c[2]).unsqueeze(0), '')
