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
    view1['camera_pose'] = torch.bmm(torch.bmm(R, T), view1['camera_pose'])
    view1['pts3d'] = geotrf(inv(view1['camera_pose']), view1['pts3d'])
    view1 = random_mask(view1)
    batch = [view1, view1]
    return loss_of_one_batch_with_single_view(batch, model, criterion, device, symmetrize_batch, use_amp, ret)

def random_translation_matrix_3d(batch_size=1, scale=1.0):
    """
    output:
    - T.shape:  (batch_size, 4, 4)
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
    output:
    - R.shape: (batch_size, 4, 4) 
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
    B = pts3d.shape[0]
    # random change the pts3d's rotate and translate
    R = random_rotation_matrix().to(pts3d.device)
    if scale is None:
        scale = pts3d.max() - pts3d.min()
    T = random_translation_matrix_3d(scale=scale).to(pts3d.device)
    # change the pts3d
    pts3d = geotrf(R, pts3d)
    pts3d = geotrf(T, pts3d)
    #TODO: change to individual value for each matrix
    return pts3d, R.repeat(B,1,1), T.repeat(B,1,1)

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