# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training code for DUSt3R
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from dust3r.training import build_dataset, save_final_model
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.model import AsymmetricCroCo3DStereo, inf  # noqa: F401, needed when loading the model
from dust3r_inpaint.model import DUSt3R_InpaintModel
from dust3r.datasets import get_data_loader  # noqa
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch
from dust3r_inpaint.inference import loss_of_one_batch_with_single_view, loss_of_one_batch_with_random_transform  # noqa

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa
from safetensors.torch import load_file
import open3d as o3d
from PIL import Image
import copy


def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume
    last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
    #TODO: change to auto resume
    args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    # training dataset and loader
    print('Building train dataset {:s}'.format(args.train_dataset))
    #  dataset and loader
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)
    print('Building test dataset {:s}'.format(args.train_dataset))
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    # model
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model)
    print(f'>> Creating train criterion = {args.train_criterion}')
    train_criterion = eval(args.train_criterion).to(device)
    print(f'>> Creating test criterion = {args.test_criterion or args.train_criterion}')
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.pretrained and not args.resume:
        print('Loading pretrained: ', args.pretrained)
        if args.pretrained.endswith('.pth'):
            ckpt = torch.load(args.pretrained, map_location=device)
            print(model.load_state_dict(ckpt['model'], strict=False))
            del ckpt  # in case it occupies memory
        elif args.pretrained.endswith('.safetensors'):
            state_dict = load_file(args.pretrained)
            model.load_state_dict(state_dict, strict=False)
            del state_dict  # in case it occupies memory

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update({test_name + '_' + k: v for k, v in test_stats[test_name].items()})

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, fname=fname, best_so_far=best_so_far)

    best_so_far = misc.load_model(args=args, model_without_ddp=model_without_ddp,
                                  optimizer=optimizer, loss_scaler=loss_scaler)
    
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model parameters: {total_params}")
    if best_so_far is None:
        best_so_far = float('inf')
    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.start_epoch, args.epochs + 1):

        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs:
                save_model(epoch - 1, 'last', best_so_far)

        # Test on multiple datasets
        new_best = False
        if (epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0):
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                stats = test_one_epoch(model, test_criterion, testset,
                                       device, epoch, log_writer=log_writer, args=args, prefix=test_name)
                test_stats[test_name] = stats

                # Save best of all
                if stats['loss_med'] < best_so_far:
                    best_so_far = stats['loss_med']
                    new_best = True

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch - 1, 'best', best_so_far)
        if epoch >= args.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model, train_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    save_final_model(args, args.epochs, model_without_ddp, best_so_far=best_so_far)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args, log_writer=None, vis_interval=500, cache_dir_root = 'cache'):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)
        # batch: view1(img(B,C,H,W), pts3d(B,3,N), camera_pose(B,4,4,...)), view2(...)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)
        #TIP: change with random transform
        
        origin_view1 = copy.deepcopy(batch[0])
        result = loss_of_one_batch_with_random_transform(batch, model, criterion, device,
                                                          symmetrize_batch=False,
                                                          use_amp=bool(args.amp))
        view1, view2, pred1, pred2, loss_tuple = result['view1'], result['view2'], result['pred1'], result['pred2'], result['loss']
        loss, loss_details = loss_tuple  # criterion returns two values
        loss_value = float(loss)
        
        if data_iter_step % vis_interval == 0:
            # log visual sample:
            with torch.no_grad():
                cache_dir = os.path.join(args.output_dir  , cache_dir_root, f'epoch-{epoch}_step-{data_iter_step}')
                os.makedirs(cache_dir, exist_ok=True)
                
                mask_begin_pos = view1['mask_begin_pos']
                mask_end_pos = view1['mask_end_pos']
                x1,y1 = mask_begin_pos
                x2,y2 = mask_end_pos
                
                # save img
                img = view1['img'][0].permute(1,2,0)
                img = ((img - img.min())/(img.max()-img.min())*255).cpu().numpy().astype(np.uint8)
                Image.fromarray(img).save(f'{cache_dir}/({x1},{y1})-({x2},{y2})img.jpg')
                masked_img = img
                masked_img[x1:x2,y1:y2,:] = 0
                Image.fromarray(masked_img).save(f'{cache_dir}/({x1},{y1})-({x2},{y2})masked_img.jpg')
                
                # save pcd
                origin_pcd = geotrf(inv(origin_view1['camera_pose']), origin_view1['pts3d'])
                new_no_masked_pcd = geotrf(inv(view1['camera_pose']), view1['pts3d'])
                new_pcd = view1['camera_pts3d']
                # out_pcd = pred1['pts3d']
                _, _, out_pcd, _, valid1, _, _ = get_all_pts3d(view1, view2, pred1, pred2)
                valid1[1:,:,:]=0
                origin_pcd = normalize_pointcloud(origin_pcd, None, 'avg_dis')
                new_no_masked_pcd = normalize_pointcloud(new_no_masked_pcd, None, 'avg_dis')
                new_pcd = normalize_pointcloud(new_pcd, None, 'avg_dis')
                save_pcd(f'{cache_dir}/({x1},{y1})-({x2},{y2})out_pcd_no_norm.ply', out_pcd[valid1])
                out_pcd = normalize_pointcloud(out_pcd, None, 'avg_dis')
                
                # save the first one
                save_pcd(f'{cache_dir}/({x1},{y1})-({x2},{y2})origin_pcd.ply', origin_pcd[0])
                save_pcd(f'{cache_dir}/({x1},{y1})-({x2},{y2})new_no_masked_pcd.ply', new_no_masked_pcd[0])
                save_pcd(f'{cache_dir}/({x1},{y1})-({x2},{y2})new_pcd.ply', new_pcd[0])
                save_pcd(f'{cache_dir}/({x1},{y1})-({x2},{y2})out_pcd.ply', out_pcd[valid1])
                masked_origin_pcd = origin_pcd
                masked_origin_pcd[:,x1:x2,y1:y2,:] = 0
                save_pcd(f'{cache_dir}/({x1},{y1})-({x2},{y2})masked_origin_pcd.ply', masked_origin_pcd[0])
                
                # save R and T info
                R = view1['R'][0]
                T = view1['T'][0]
                with open(f'{cache_dir}/({x1},{y1})-({x2},{y2})RT.txt', 'w') as f:
                    f.write(f'R:\n{R}\n')
                    f.write(f'T:\n{T}\n')
                

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step + 1) % accum_iter == 0 and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            loss_value_reduce = misc.all_reduce_mean(loss_value)  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_lr', lr, epoch_1000x)
            log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)
            for name, val in loss_details.items():
                log_writer.add_scalar('train_' + name, val, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test', save_pts3d_interval=5, cache_dir='cache/test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    for idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        results = loss_of_one_batch_with_random_transform(batch, model, criterion, device,
                                       symmetrize_batch=False,
                                       use_amp=bool(args.amp))
        loss_value, loss_details = results['loss']  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + '_' + name, val, 1000 * epoch)

    return results

def save_pcd(save_path, points):
    if isinstance(points, torch.Tensor):
        if points.ndim > 2:
            points = points.flatten(0,-2)
        points = points.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(save_path, pcd)

@torch.no_grad()
def test_one_epoch_with_single_view(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test', save_pts3d_interval=5, cache_dir='cache/test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Single View Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    for idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        results = loss_of_one_batch_with_single_view(batch, model, criterion, device,
                                       use_amp=bool(args.amp))
        loss_value, loss_details = results['loss']  # criterion returns two values
        view1,view2 = results['view1'],results['view2']
        pred1,pred2 = results['pred1'],results['pred2']
        metric_logger.update(loss=float(loss_value), **loss_details)
        if idx % save_pts3d_interval == 0:
            #save as npz
            os.makedirs(cache_dir, exist_ok=True)
            # save img
            img = results['view1']['img'][0].permute(1,2,0)
            img = ((img - img.min())/(img.max()-img.min())*255).cpu().numpy().astype(np.uint8)
            Image.fromarray(img).save(os.path.join(cache_dir, f'{idx}.jpg'))
            gt_pts1, _, pr_pts1, _, _, _, _ = get_all_pts3d(view1, view2, pred1, pred2)
            save_pcd(os.path.join(cache_dir, f'{idx}-single-gt.ply'), gt_pts1)
            save_pcd(os.path.join(cache_dir, f'{idx}-single-pred-loss{loss_value:0.5f}.ply'), pr_pts1)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + '_' + name, val, 1000 * epoch)

    return results

@torch.no_grad()
def test_one_epoch2(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test', save_pts3d_interval=5, cache_dir='cache/test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    for idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        results = loss_of_one_batch(batch, model, criterion, device,
                                       use_amp=bool(args.amp))
        loss_value, loss_details = results['loss']  # criterion returns two values
        view1,view2 = results['view1'],results['view2']
        pred1,pred2 = results['pred1'],results['pred2']
        metric_logger.update(loss=float(loss_value), **loss_details)
        if idx % save_pts3d_interval == 0:
            #save as npz
            os.makedirs(cache_dir, exist_ok=True)
            gt_pts1, _, pr_pts1, _, _, _, _ = get_all_pts3d(view1, view2, pred1, pred2)
            save_pcd(os.path.join(cache_dir, f'{idx}-gt.ply'), gt_pts1)
            save_pcd(os.path.join(cache_dir, f'{idx}-pred-loss{loss_value:0.5f}.ply'), pr_pts1)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + '_' + name, val, 1000 * epoch)

    return results

def test(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume
    last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
    args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    # training dataset and loader
    print('Building train dataset {:s}'.format(args.train_dataset))
    #  dataset and loader
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)
    print('Building test dataset {:s}'.format(args.train_dataset))
    data_loader_test = [build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')][0]

    # model
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model)
    print(f'>> Creating train criterion = {args.train_criterion}')
    train_criterion = eval(args.train_criterion).to(device)
    print(f'>> Creating test criterion = {args.test_criterion or args.train_criterion}')
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.pretrained and not args.resume:
        print('Loading pretrained: ', args.pretrained)
        if args.pretrained.endswith('.pth'):
            ckpt = torch.load(args.pretrained, map_location=device)
            print(model.load_state_dict(ckpt['model'], strict=False))
            del ckpt  # in case it occupies memory
        elif args.pretrained.endswith('.safetensors'):
            state_dict = load_file(args.pretrained)
            model.load_state_dict(state_dict, strict=False)
            del state_dict  # in case it occupies memory

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    best_so_far = misc.load_model(args=args, model_without_ddp=model_without_ddp,
                                  optimizer=optimizer, loss_scaler=loss_scaler)
    if best_so_far is None:
        best_so_far = float('inf')
    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    test_stats = {}
    # for test_name, testset in data_loader_test.items():
    test_one_epoch_with_single_view(model, test_criterion, data_loader_test,
                            device, 0, log_writer=log_writer, args=args, save_pts3d_interval=20)
    test_one_epoch2(model, test_criterion, data_loader_test,
                            device, 0, log_writer=log_writer, args=args, save_pts3d_interval=20)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Test time {}'.format(total_time_str))