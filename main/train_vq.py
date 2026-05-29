#!/usr/bin/env python
"""
bash scripts/train.sh CodeTalkerV2_s1 config/vocaset/stage1.yaml vocaset s1
bash scripts/<train.sh|test.sh> <exp_name> config/<vocaset|BIWI>/<stage1|stage2>.yaml <vocaset|BIWI> <s1|s2>
"""

import os
import time

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from base.baseTrainer import poly_learning_rate, reduce_tensor, save_checkpoint
from base.utilities import AverageMeter, get_logger, get_parser, main_process
from metrics.loss import calc_vq_loss
from models import get_model
from utils.indices_util import get_region_indices

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def main(cfg_path, opts=None):
    args = get_parser(cfg_path, opts)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.train_gpu = args.train_gpu[0]
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    cfg = args
    cfg.gpu = gpu

    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url, world_size=cfg.world_size,
                                rank=cfg.rank)
    
    # ####################### 区域索引准备 ####################### #
    region_indices = get_region_indices()

    # ####################### Model ####################### #
    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(cfg.save_path)
    model = get_model(cfg)
    if cfg.sync_bn:
        logger.info("using DDP synced BN")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if main_process(cfg):
        logger.info(cfg)
        logger.info("=> creating model ...")
        # model.summary(logger, writer)
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
        cfg.batch_size_val = int(cfg.batch_size_val / ngpus_per_node)
        cfg.workers = int(cfg.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(gpu), device_ids=[gpu])
    else:
        torch.cuda.set_device(gpu)
        model = model.cuda()

    # ####################### Optimizer ####################### #
    if cfg.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.base_lr, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr)

    if cfg.StepLR:
        scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    else:
        scheduler = None

    # ####################### Data Loader ####################### #
    from dataset.data_loader import get_dataloaders
    dataset = get_dataloaders(cfg)
    train_loader = dataset['train']
    train_sampler = dataset['train_sampler']  # None for single-GPU
    if cfg.evaluate:
        val_loader = dataset['valid']

    # ####################### Train ############################# #
    for epoch in range(cfg.start_epoch, cfg.epochs):
        # DDP: 每 epoch 重新 shuffle，保证各卡看到不同数据
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # 用 split_loss 接收分区域损失
        rec_loss_train, quant_loss_train, pp_train, split_loss = train(train_loader, model, calc_vq_loss, optimizer, epoch, cfg, region_indices)
        lip_loss, eye_loss, other_loss = split_loss
        epoch_log = epoch + 1
        if cfg.StepLR:
            scheduler.step()
        if main_process(cfg):
            logger.info('TRAIN Epoch: {} '
                        'loss_train: {} '
                        'pp_train: {} '
                        .format(epoch_log, rec_loss_train, pp_train)
                        )
            for m, s in zip([rec_loss_train, quant_loss_train, pp_train],
                            ["train/rec_loss", "train/quant_loss", "train/perplexity"]):
                writer.add_scalar(s, m, epoch_log)
            # 写入分区域损失
            writer.add_scalar("train/regions/lip_loss", lip_loss, epoch_log)
            writer.add_scalar("train/regions/eye_loss", eye_loss, epoch_log)
            writer.add_scalar("train/regions/other_loss", other_loss, epoch_log)

        if cfg.evaluate and (epoch_log % cfg.eval_freq == 0):
            rec_loss_val, quant_loss_val, pp_val = validate(val_loader, model, calc_vq_loss, epoch, cfg)
            if main_process(cfg):
                logger.info('VAL Epoch: {} '
                            'loss_val: {} '
                            'pp_val: {} '
                            .format(epoch_log, rec_loss_val, pp_val)
                            )
                for m, s in zip([rec_loss_val, quant_loss_val, pp_val],
                                ["val/rec_loss", "val/quant_loss", "val/perplexity"]):
                    writer.add_scalar(s, m, epoch_log)


        if (epoch_log % cfg.save_freq == 0) and main_process(cfg):
            save_checkpoint(model,
                            sav_path=os.path.join(cfg.save_path, 'model')
                            )


def train(train_loader, model, loss_fn, optimizer, epoch, cfg, region_indices):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    rec_loss_meter = AverageMeter()
    quant_loss_meter = AverageMeter()
    pp_meter = AverageMeter()

    lip_region, eye_region, other_region = region_indices
    # ####################### 新增：为三个区域损失创建记录器 ############################# #
    lip_meter, eye_meter, other_meter = AverageMeter(), AverageMeter(), AverageMeter() 

    model.train()
    end = time.time()
    max_iter = cfg.epochs * len(train_loader)
    for i, (data, _, template, _, _) in enumerate(train_loader):
        current_iter = epoch * len(train_loader) + i + 1
        data_time.update(time.time() - end)
        data = data.cuda(cfg.gpu, non_blocking=True)
        template = template.cuda(cfg.gpu, non_blocking=True)

        out, quant_loss, info, loss_zero = model(data, template)
        lip_vertices, eye_vertices, other_vertices = info[1] # 从新的 info 结构中取出分区域结果

        gt_vertices = data.view(data.shape[0], data.shape[1], -1, 3)
        lip_vertices = lip_vertices.view(lip_vertices.shape[0], lip_vertices.shape[1], -1, 3)
        eye_vertices = eye_vertices.view(eye_vertices.shape[0], eye_vertices.shape[1], -1, 3)
        other_vertices = other_vertices.view(other_vertices.shape[0], other_vertices.shape[1], -1, 3)

        loss_lip = nn.L1Loss()(lip_vertices[:, :, lip_region], gt_vertices[:, :, lip_region])
        loss_eye = nn.L1Loss()(eye_vertices[:, :, eye_region], gt_vertices[:, :, eye_region])
        loss_other = nn.L1Loss()(other_vertices[:, :, other_region], gt_vertices[:, :, other_region])

        rec_loss = loss_lip * 2.0 + loss_eye * 1.0 + loss_other * 1.0  # 给嘴部区域更大的权重
        # 直接对全脸输出加 L1，确保三个 decoder 的加和结果收敛到真值
        loss_full = nn.L1Loss()(out, data.view(data.shape[0], data.shape[1], -1))
        train_loss = loss_full + rec_loss + cfg.quant_loss_weight * quant_loss.mean() + loss_zero * cfg.loss_zero_weight
        _, loss_details = loss_fn(out, data, quant_loss, quant_loss_weight=cfg.quant_loss_weight)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        for m, x in zip([rec_loss_meter, quant_loss_meter, pp_meter],
                        [loss_details[0], loss_details[1], info[0]]): #info[0] is perplexity
            m.update(x.item(), 1)
        
        # Adjust lr
        if cfg.poly_lr:
            current_lr = poly_learning_rate(cfg.base_lr, current_iter, max_iter, power=cfg.power)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % cfg.print_freq == 0 and main_process(cfg):
            # logger.info('Epoch: [{}/{}][{}/{}] '
            #             'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
            #             'Batch: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
            #             'Remain: {remain_time} '
            #             'Loss: {loss_meter.val:.4f} '
            #             .format(epoch + 1, cfg.epochs, i + 1, len(train_loader),
            #                     batch_time=batch_time, data_time=data_time,
            #                     remain_time=remain_time,
            #                     loss_meter=rec_loss_meter
            #                     ))
            for m, s in zip([rec_loss_meter, quant_loss_meter],
                            ["train_batch/rec_loss", "train_batch/quant_loss"]):
                writer.add_scalar(s, m.val, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)

        # 记录分区域loss
        lip_meter.update(loss_lip.item(), data.size(0))
        eye_meter.update(loss_eye.item(), data.size(0))
        other_meter.update(loss_other.item(), data.size(0))

    return rec_loss_meter.avg, quant_loss_meter.avg, pp_meter.avg, (lip_meter.avg, eye_meter.avg, other_meter.avg)


def validate(val_loader, model, loss_fn, epoch, cfg):
    rec_loss_meter = AverageMeter()
    quant_loss_meter = AverageMeter()
    pp_meter = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (data, _, template, _, _) in enumerate(val_loader):
            data = data.cuda(cfg.gpu, non_blocking=True)
            template = template.cuda(cfg.gpu, non_blocking=True)

            out, quant_loss, info, _ = model(data, template)

            # LOSS
            loss, loss_details = loss_fn(out, data, quant_loss, quant_loss_weight=cfg.quant_loss_weight)

            if cfg.distributed:
                loss = reduce_tensor(loss, cfg)


            for m, x in zip([rec_loss_meter, quant_loss_meter, pp_meter],
                            [loss_details[0], loss_details[1], info[0]]):
                m.update(x.item(), 1) #batch_size = 1 for validation


    return rec_loss_meter.avg, quant_loss_meter.avg, pp_meter.avg


if __name__ == '__main__':
    main() 
    