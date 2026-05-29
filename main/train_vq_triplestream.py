"""
Route B: 真三流 VQ-AE 训练脚本。

损失设计：
  - 每条流只计算自己区域的 L1 重建损失（ground truth 同样只取该区域的顶点偏移量）
  - 三路损失加权求和：lip × cfg.lip_weight (默认 2.0)，eye/other × 1.0
  - 无"全脸 L1"项（全脸重建是三路之和，用全脸 L1 会引入区域间的梯度耦合）
  - VQ 损失：三码本 VQ loss 之和

用法:
  python run_vq_triplestream.py [--exp_name NAME] [--gpus 0]
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
from models import get_model

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

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(cfg.save_path)
    model = get_model(cfg)
    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if main_process(cfg):
        logger.info(cfg)
        logger.info("=> creating model (Route B: triple-stream, truly independent regions)")
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
            train_loader, model, optimizer, epoch, cfg)
        lip_loss, eye_loss, other_loss = split_loss
        epoch_log = epoch + 1

        if cfg.StepLR:
            scheduler.step()

        if main_process(cfg):
            logger.info(f'TRAIN Epoch: {epoch_log}  rec_loss: {rec_loss_train:.5f}  pp: {pp_train:.2f}')
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
                logger.info(f'VAL Epoch: {epoch_log}  rec_loss: {rec_loss_val:.5f}  pp: {pp_val:.2f}')
                for val, tag in zip(
                    [rec_loss_val, quant_loss_val, pp_val],
                    ["val/rec_loss", "val/quant_loss", "val/perplexity"]
                ):
                    writer.add_scalar(tag, val, epoch_log)

        if (epoch_log % cfg.save_freq == 0) and main_process(cfg):
            save_checkpoint(model, sav_path=os.path.join(cfg.save_path, 'model'))


def _get_region_bufs(model):
    """从 DDP 包装或直接模型中获取区域索引 buffer。"""
    m = model.module if hasattr(model, 'module') else model
    return m.lip_region, m.eye_region, m.other_region, m.lip_coords, m.eye_coords, m.other_coords


def train(train_loader, model, optimizer, epoch, cfg):
    rec_loss_meter   = AverageMeter()
    quant_loss_meter = AverageMeter()
    pp_meter         = AverageMeter()
    lip_meter        = AverageMeter()
    eye_meter        = AverageMeter()
    other_meter      = AverageMeter()

    lip_w   = getattr(cfg, 'lip_weight',   2.0)
    eye_w   = getattr(cfg, 'eye_weight',   1.0)
    other_w = getattr(cfg, 'other_weight', 1.0)

    model.train()
    end = time.time()
    max_iter = cfg.epochs * len(train_loader)

    lip_region = eye_region = other_region = None  # 延迟初始化（需要 model 在 GPU 上）

    for i, (data, _, template, _, _) in enumerate(train_loader):
        current_iter = epoch * len(train_loader) + i + 1
        data     = data.cuda(cfg.gpu, non_blocking=True)
        template = template.cuda(cfg.gpu, non_blocking=True)

        # 懒获取区域 buffer（在 GPU 上，避免每步从 CPU 传）
        if lip_region is None:
            lip_region, eye_region, other_region, lip_coords, eye_coords, other_coords = \
                _get_region_bufs(model)

        full_vertices, emb_loss, info = model(data, template)
        # info = (perplexity, (dec_lip, dec_eye, dec_other), indices)
        dec_lip, dec_eye, dec_other = info[1]

        # 每流的 GT offset（模型内部已减 template，对应的 GT 也用 offset 比较）
        B, T = data.shape[:2]
        x_offset = data.view(B, T, -1) - template.unsqueeze(1)

        gt_lip   = x_offset[:, :, lip_coords]    # [B, T, N_lip*3]
        gt_eye   = x_offset[:, :, eye_coords]
        gt_other = x_offset[:, :, other_coords]

        loss_lip   = nn.L1Loss()(dec_lip,   gt_lip)
        loss_eye   = nn.L1Loss()(dec_eye,   gt_eye)
        loss_other = nn.L1Loss()(dec_other, gt_other)

        # 加权区域损失 + VQ 损失（无全脸 L1，保持流间梯度独立）
        train_loss = (lip_w * loss_lip + eye_w * loss_eye + other_w * loss_other
                      + cfg.quant_loss_weight * emb_loss.mean())

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if cfg.poly_lr:
            current_lr = poly_learning_rate(cfg.base_lr, current_iter, max_iter, power=cfg.power)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # 记录：rec_loss 用全脸 L1（便于与 baseline WandB 对比）
        loss_full = nn.L1Loss()(full_vertices.view(B, T, -1), data.view(B, T, -1))
        rec_loss_meter.update(loss_full.item(), 1)
        quant_loss_meter.update(emb_loss.mean().item(), 1)
        pp_meter.update(info[0].item(), 1)
        lip_meter.update(loss_lip.item(), data.size(0))
        eye_meter.update(loss_eye.item(), data.size(0))
        other_meter.update(loss_other.item(), data.size(0))

        end = time.time()

        if (i + 1) % cfg.print_freq == 0 and main_process(cfg):
            writer.add_scalar('train_batch/rec_loss',   rec_loss_meter.val,   current_iter)
            writer.add_scalar('train_batch/quant_loss', quant_loss_meter.val, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)

    return (rec_loss_meter.avg, quant_loss_meter.avg, pp_meter.avg,
            (lip_meter.avg, eye_meter.avg, other_meter.avg))


def validate(val_loader, model, cfg):
    rec_loss_meter   = AverageMeter()
    quant_loss_meter = AverageMeter()
    pp_meter         = AverageMeter()
    model.eval()

    lip_region = eye_region = other_region = None

    with torch.no_grad():
        for data, _, template, _, _ in val_loader:
            data     = data.cuda(cfg.gpu, non_blocking=True)
            template = template.cuda(cfg.gpu, non_blocking=True)

            full_vertices, emb_loss, info = model(data, template)

            loss_full = nn.L1Loss()(full_vertices.view(data.shape[0], data.shape[1], -1),
                                    data.view(data.shape[0], data.shape[1], -1))
            if cfg.distributed:
                loss_full = reduce_tensor(loss_full, cfg)

            rec_loss_meter.update(loss_full.item(), 1)
            quant_loss_meter.update(emb_loss.mean().item(), 1)
            pp_meter.update(info[0].item(), 1)

    return rec_loss_meter.avg, quant_loss_meter.avg, pp_meter.avg


if __name__ == '__main__':
    main()
