# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import tqdm
import torch
import math
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf, inv
from dust3r.inference import loss_of_one_batch

def loss_of_one_batch_with_single_view(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None):
    view1, _ = batch
    batch = [view1, view1]
    return loss_of_one_batch(batch, model, criterion, device, symmetrize_batch, use_amp, ret)

def loss_of_one_batch_with_random_transform(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None):
    view1, _ = batch
    _, R, T = random_transform(view1['pts3d'])
    view1['camera_pose'] = torch.mm(torch.mm(R, T), view1['camera_pose'])
    view1['pts3d'] = geotrf(inv(view1['camera_pose']), view1['pts3d'])
    view1 = random_mask(view1)
    batch = [view1, view1]
    return loss_of_one_batch_with_single_view(batch, model, criterion, device, symmetrize_batch, use_amp, ret)

def random_translation_matrix_3d(scale=1.0):
    tx = (torch.rand(1) - 0.5) * scale
    ty = (torch.rand(1) - 0.5) * scale
    tz = (torch.rand(1) - 0.5) * scale

    T = torch.tensor([[1, 0, 0, tx],
                      [0, 1, 0, ty],
                      [0, 0, 1, tz],
                      [0, 0, 0, 1]])

    return T


def random_rotation_matrix():
    theta = torch.rand(1) * 2 * math.pi 
    phi = torch.rand(1) * math.pi  
    psi = torch.rand(1) * 2 * math.pi  

    Rz = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                       [torch.sin(theta), torch.cos(theta), 0],
                       [0, 0, 1]])

    Ry = torch.tensor([[torch.cos(phi), 0, torch.sin(phi)],
                       [0, 1, 0],
                       [-torch.sin(phi), 0, torch.cos(phi)]])

    Rx = torch.tensor([[1, 0, 0],
                       [0, torch.cos(psi), -torch.sin(psi)],
                       [0, torch.sin(psi), torch.cos(psi)]])
    R = torch.mm(torch.mm(Rz, Ry), Rx)
    # to 4x4
    R = torch.cat((R, torch.tensor([[0, 0, 0, 1]])), dim=0)
    return R

def random_transform(pts3d,scale=None):
    # random change the pts3d's rotate and translate
    R = random_rotation_matrix().to(pts3d.device)
    if scale is None:
        scale = pts3d.max() - pts3d.min()
    T = random_translation_matrix_3d(scale=scale).to(pts3d.device)
    # change the pts3d
    pts3d = geotrf(R, pts3d)
    pts3d = geotrf(T, pts3d)
    return pts3d, R, T

def random_mask(view,min_scale=10,max_scale=200):
    img = view['img']
    width, height = view['img'].size
    mask_center = torch.rand(2) * torch.tensor([width,height])
    min_scale = torch.tensor([min_scale,min_scale])
    max_scale = torch.tensor([max_scale,max_scale])
    delta_begin = (torch.rand(2) * (max_scale - min_scale) + min_scale) * 0.5
    delta_end = (torch.rand(2) * (max_scale - min_scale) + min_scale) * 0.5
    mask_begin_pos = (mask_center - delta_begin).clamp(min=0).to(torch.int)
    mask_end_pos = (mask_center + delta_end).minimum(torch.tensor([width,height])).to(torch.int)
    x1,y1 = mask_begin_pos
    x2,y2 = mask_end_pos
    mask = torch.zeros_like(img)
    mask[...,x1:x2,y1:y2] = 1
    mask = mask.permute(0,3,1,2)
    masked_pts3d = view['pts3d']
    masked_pts3d[:,x1:x2,y1:y2,...] = 0
    masked_pts3d = masked_pts3d.permute(0,3,1,2)
    view['input'] = torch.cat((img,masked_pts3d,mask),dim=1)
    view['mask_begin_pos'] = mask_begin_pos
    view['mask_end_pos'] = mask_end_pos
    return view