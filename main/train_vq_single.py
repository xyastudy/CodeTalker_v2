"""
Route A: 单码本 VQ-AE 训练脚本。

与原始 baseline 的唯一差异：在全脸 L1 基础上叠加区域加权重建损失。
  - lip 区域权重 × cfg.lip_weight（默认 2.0）
  - eye/other 权重 × 1.0

用法:
  python run_vq_single.py [--exp_name NAME] [--gpus 0]
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
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                world_size=cfg.world_size, rank=cfg.rank)

    region_indices = get_region_indices()

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(cfg.save_path)
    model = get_model(cfg)
    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if main_process(cfg):
        logger.info(cfg)
        logger.info("=> creating model (Route A: single codebook + region-weighted loss)")
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
        cfg.batch_size_val = int(cfg.batch_size_val / ngpus_per_node)
        cfg.workers = int(cfg.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(gpu), device_ids=[gpu])
    else:
        torch.cuda.set_device(gpu)
        model = model.cuda()

    if cfg.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.base_lr,
                                    momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr)

    scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma) if cfg.StepLR else None

    from dataset.data_loader import get_dataloaders
    dataset = get_dataloaders(cfg)
    train_loader = dataset['train']
    train_sampler = dataset['train_sampler']  # None for single-GPU
    val_loader = dataset['valid'] if cfg.evaluate else None

    for epoch in range(cfg.start_epoch, cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        rec_loss_train, quant_loss_train, pp_train, split_loss = train(
            train_loader, model, optimizer, epoch, cfg, region_indices)
        lip_loss, eye_loss, other_loss = split_loss
        epoch_log = epoch + 1

        if cfg.StepLR:
            scheduler.step()

        if main_process(cfg):
            logger.info(f'TRAIN Epoch: {epoch_log}  rec_loss: {rec_loss_train:.2e}  pp: {pp_train:.2f}')
            for val, tag in zip(
                [rec_loss_train, quant_loss_train, pp_train],
                ["train/rec_loss", "train/quant_loss", "train/perplexity"]
            ):
                writer.add_scalar(tag, val, epoch_log)
            writer.add_scalar("train/regions/lip_loss",   lip_loss,   epoch_log)
            writer.add_scalar("train/regions/eye_loss",   eye_loss,   epoch_log)
            writer.add_scalar("train/regions/other_loss", other_loss, epoch_log)

        if cfg.evaluate and val_loader and (epoch_log % cfg.eval_freq == 0):
            rec_loss_val, quant_loss_val, pp_val = validate(val_loader, model, cfg)
            if main_process(cfg):
                logger.info(f'VAL Epoch: {epoch_log}  rec_loss: {rec_loss_val:.2e}  pp: {pp_val:.2f}')
                for val, tag in zip(
                    [rec_loss_val, quant_loss_val, pp_val],
                    ["val/rec_loss", "val/quant_loss", "val/perplexity"]
                ):
                    writer.add_scalar(tag, val, epoch_log)

        if (epoch_log % cfg.save_freq == 0) and main_process(cfg):
            save_checkpoint(model, sav_path=os.path.join(cfg.save_path, 'model'))


def train(train_loader, model, optimizer, epoch, cfg, region_indices):
    rec_loss_meter  = AverageMeter()
    quant_loss_meter = AverageMeter()
    pp_meter        = AverageMeter()
    lip_meter       = AverageMeter()
    eye_meter       = AverageMeter()
    other_meter     = AverageMeter()

    lip_region, eye_region, other_region = region_indices
    lip_w   = getattr(cfg, 'lip_weight',   2.0)
    eye_w   = getattr(cfg, 'eye_weight',   1.0)
    other_w = getattr(cfg, 'other_weight', 1.0)

    model.train()
    end = time.time()
    max_iter = cfg.epochs * len(train_loader)

    for i, (data, _, template, _, _) in enumerate(train_loader):
        current_iter = epoch * len(train_loader) + i + 1
        data     = data.cuda(cfg.gpu, non_blocking=True)
        template = template.cuda(cfg.gpu, non_blocking=True)

        out, quant_loss, info = model(data, template)
        # info[0] = perplexity

        # 区域加权重建损失：lip/eye/other 三块合起来覆盖全脸，不再单独加全脸 L1 避免重复计算
        B, T = data.shape[:2]
        out_3d  = out.view(B, T, -1, 3)
        gt_3d   = data.view(B, T, -1, 3)

        loss_lip   = nn.L1Loss()(out_3d[:, :, lip_region],   gt_3d[:, :, lip_region])
        loss_eye   = nn.L1Loss()(out_3d[:, :, eye_region],   gt_3d[:, :, eye_region])
        loss_other = nn.L1Loss()(out_3d[:, :, other_region], gt_3d[:, :, other_region])
        loss_region = lip_w * loss_lip + eye_w * loss_eye + other_w * loss_other

        train_loss = loss_region + cfg.quant_loss_weight * quant_loss.mean()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # 学习率调整
        if cfg.poly_lr:
            current_lr = poly_learning_rate(cfg.base_lr, current_iter, max_iter, power=cfg.power)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # 记录全脸 L1（与 baseline/val 量级一致，方便 WandB 对比）
        with torch.no_grad():
            loss_full = nn.L1Loss()(out, data.view(B, T, -1))
        rec_loss_meter.update(loss_full.item(), 1)
        quant_loss_meter.update(quant_loss.mean().item(), 1)
        pp_meter.update(info[0].item(), 1)
        lip_meter.update(loss_lip.item(), data.size(0))
        eye_meter.update(loss_eye.item(), data.size(0))
        other_meter.update(loss_other.item(), data.size(0))

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * (time.time() - end)
        end = time.time()

        if (i + 1) % cfg.print_freq == 0 and main_process(cfg):
            writer.add_scalar('train_batch/rec_loss',   rec_loss_meter.val,   current_iter)
            writer.add_scalar('train_batch/quant_loss', quant_loss_meter.val, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)

    return (rec_loss_meter.avg, quant_loss_meter.avg, pp_meter.avg,
            (lip_meter.avg, eye_meter.avg, other_meter.avg))


def validate(val_loader, model, cfg):
    rec_loss_meter  = AverageMeter()
    quant_loss_meter = AverageMeter()
    pp_meter        = AverageMeter()
    model.eval()

    with torch.no_grad():
        for data, _, template, _, _ in val_loader:
            data     = data.cuda(cfg.gpu, non_blocking=True)
            template = template.cuda(cfg.gpu, non_blocking=True)

            out, quant_loss, info = model(data, template)
            loss_full = nn.L1Loss()(out, data.view(data.shape[0], data.shape[1], -1))

            if cfg.distributed:
                loss_full = reduce_tensor(loss_full, cfg)

            rec_loss_meter.update(loss_full.item(), 1)
            quant_loss_meter.update(quant_loss.mean().item(), 1)
            pp_meter.update(info[0].item(), 1)

    return rec_loss_meter.avg, quant_loss_meter.avg, pp_meter.avg


if __name__ == '__main__':
    main()
