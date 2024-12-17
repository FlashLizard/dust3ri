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


def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2

def loss_of_one_batch(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None):
    view1, view2 = batch
    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    if symmetrize_batch:
        view1, view2 = make_batch_symmetric(batch)

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        pred1, pred2 = model(view1, view2)

        # loss is supposed to be symmetric
        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(view1, view2, pred1, pred2) if criterion is not None else None

    result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=loss)
    return result[ret] if ret else result

@torch.no_grad()
def inference(pairs, model, device, batch_size=8, verbose=True):
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch(collate_with_cat(pairs[i:i + batch_size]), model, None, device)
        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    return result


def check_if_same_size(pairs):
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)


def get_pred_pts3d(gt, pred, use_pose=False):
    if 'depth' in pred and 'pseudo_focal' in pred:
        try:
            pp = gt['camera_intrinsics'][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif 'pts3d' in pred:
        # pts3d from my camera
        pts3d = pred['pts3d']

    elif 'pts3d_in_other_view' in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred['pts3d_in_other_view']  # return!

    if use_pose:
        camera_pose = pred.get('camera_pose')
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(gt_pts1, gt_pts2, pr_pts1, pr_pts2=None, fit_mode='weiszfeld_stop_grad', valid1=None, valid2=None):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    # concat the pointcloud
    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None

    all_gt = torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1) if gt_pts2 is not None else nan_gt_pts1
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith('avg'):
        # scaling = (all_pr / all_gt).view(B, -1).mean(dim=1)
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith('median'):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith('weiszfeld'):
        # init scaling with l2 closed form
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip_(min=1e-8).reciprocal()
            # update the scaling with the new weights
            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f'bad {fit_mode=}')

    if fit_mode.endswith('stop_grad'):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)
    # assert scaling.isfinite().all(), bb()
    return scaling

def loss_of_one_batch_with_single_view(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None):
    view1, _ = batch
    batch = [view1, view1]
    return loss_of_one_batch(batch, model, criterion, device, symmetrize_batch, use_amp, ret)

def loss_of_one_batch_with_random_transform(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None):
    view1, _ = batch
    _, R, T = random_transform(view1['pts3d'])
    view1['camera_pose'] = torch.bmm(torch.bmm(R, T), view1['camera_pose'])
    view1['pts3d'] = geotrf(inv(view1['camera_pose']), view1['pts3d'])
    view1 = random_mask(view1)
    batch = [view1, view1]
    return loss_of_one_batch_with_single_view(batch, model, criterion, device, symmetrize_batch, use_amp, ret)

def random_translation_matrix_3d(batch_size=1, scale=1.0):
    """
    生成一个随机的 3D 平移矩阵，支持 batch_size。
    
    参数:
    - batch_size: 批量大小。
    - scale: 平移范围的缩放因子。
    
    返回:
    - T: 形状为 (batch_size, 4, 4) 的平移矩阵。
    """
    # 生成随机的平移向量，形状为 (batch_size, 3)
    t = (torch.rand(batch_size, 3) - 0.5) * scale

    # 构建平移矩阵
    T = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)  # 初始化为单位矩阵
    T[:, 0, 3] = t[:, 0]  # 设置 tx
    T[:, 1, 3] = t[:, 1]  # 设置 ty
    T[:, 2, 3] = t[:, 2]  # 设置 tz

    return T


def random_rotation_matrix(batch_size=1):
    """
    生成一个随机的 3D 旋转矩阵，支持 batch_size。
    
    参数:
    - batch_size: 批量大小。
    
    返回:
    - R: 形状为 (batch_size, 4, 4) 的旋转矩阵。
    """
    # 生成随机的欧拉角，形状为 (batch_size, 3)
    theta = torch.rand(batch_size) * 2 * math.pi  # 绕 z 轴旋转
    phi = torch.rand(batch_size) * math.pi        # 绕 y 轴旋转
    psi = torch.rand(batch_size) * 2 * math.pi    # 绕 x 轴旋转

    # 计算旋转矩阵的元素
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
    cos_psi, sin_psi = torch.cos(psi), torch.sin(psi)

    # 构建旋转矩阵
    Rz = torch.zeros(batch_size, 4, 4)
    Rz[:, 0, 0] = cos_theta
    Rz[:, 0, 1] = -sin_theta
    Rz[:, 1, 0] = sin_theta
    Rz[:, 1, 1] = cos_theta
    Rz[:, 2, 2] = 1
    Rz[:, 3, 3] = 1

    Ry = torch.zeros(batch_size, 4, 4)
    Ry[:, 0, 0] = cos_phi
    Ry[:, 0, 2] = sin_phi
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -sin_phi
    Ry[:, 2, 2] = cos_phi
    Ry[:, 3, 3] = 1

    Rx = torch.zeros(batch_size, 4, 4)
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = cos_psi
    Rx[:, 1, 2] = -sin_psi
    Rx[:, 2, 1] = sin_psi
    Rx[:, 2, 2] = cos_psi
    Rx[:, 3, 3] = 1

    # 组合旋转矩阵
    R = torch.bmm(torch.bmm(Rz, Ry), Rx)
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
    B, C, width, height = view['img'].shape
    mask_center = torch.rand(2) * torch.tensor([width,height])
    min_scale = torch.tensor([min_scale,min_scale])
    max_scale = torch.tensor([max_scale,max_scale])
    delta_begin = (torch.rand(2) * (max_scale - min_scale) + min_scale) * 0.5
    delta_end = (torch.rand(2) * (max_scale - min_scale) + min_scale) * 0.5
    mask_begin_pos = (mask_center - delta_begin).clamp(min=0).to(torch.int)
    mask_end_pos = (mask_center + delta_end).minimum(torch.tensor([width,height])).to(torch.int)
    x1,y1 = mask_begin_pos
    x2,y2 = mask_end_pos
    mask = torch.zeros([B,1,width,height])
    mask[...,x1:x2,y1:y2] = 1
    masked_pts3d = view['pts3d']
    masked_pts3d[:,x1:x2,y1:y2,...] = 0
    masked_pts3d = masked_pts3d.permute(0,3,1,2)
    view['input'] = torch.cat((img,masked_pts3d,mask),dim=1)
    view['mask_begin_pos'] = mask_begin_pos
    view['mask_end_pos'] = mask_end_pos
    return view