"""
第一阶段（VQ-VAE）训练启动脚本。

用法示例：
  # 单卡（默认 GPU 0）
  python run_vq.py

  # 双卡
  python run_vq.py --gpus 0 1

  # 指定实验名
  python run_vq.py --exp_name my_exp

  # 从 checkpoint 续训
  python run_vq.py --resume RUN/vocaset/s1/my_exp/model/train_epoch_50.pth

  # 加载预训练权重（不续训 epoch/optimizer，只加载模型参数）
  python run_vq.py --weight checkpoint/stage1.pth
"""

import argparse
import os
import sys
from datetime import datetime

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main.train_vq import main as train


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 (VQ-VAE) training launcher")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="实验名称，默认为 triple_codebook_<时间戳>")
    parser.add_argument("--resume", type=str, default=None,
                        help="续训 checkpoint 路径（恢复 epoch、optimizer、模型权重）")
    parser.add_argument("--weight", type=str, default=None,
                        help="预训练权重路径（只加载模型参数，不恢复 epoch/optimizer）")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0],
                        help="使用的 GPU 编号，多卡用空格分隔，例如 --gpus 0 1（默认 0）")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 yaml 中的 batch_size。多卡时至少填卡数，例如双卡填 2")
    return parser.parse_args()


def main():
    args = parse_args()

    # 多卡检查：batch_size 必须 >= 卡数，否则每卡分到 0
    n_gpus = len(args.gpus)
    if args.batch_size is not None:
        effective_bs = args.batch_size
    else:
        effective_bs = 1  # yaml 默认值
    if effective_bs < n_gpus:
        raise ValueError(
            f"batch_size ({effective_bs}) 必须 >= GPU 数量 ({n_gpus})。"
            f"请加上 --batch_size {n_gpus}"
        )

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name if args.exp_name else f"triple_codebook_{now}"
    cfg_path = os.path.join(project_root, "config/vocaset/stage1.yaml")
    save_path = os.path.join(project_root, f"RUN/vocaset/s1/{exp_name}")

    # train_gpu 传字符串形式的列表，config.merge_cfg_from_list 会用 literal_eval 解析
    opts = [
        "save_path", save_path,
        "train_gpu", str(args.gpus),
    ]
    if args.batch_size is not None:
        opts += ["batch_size", str(args.batch_size)]
    if args.resume:
        opts += ["resume", args.resume]
    if args.weight:
        opts += ["weight", args.weight]

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in args.gpus)
    os.environ["OMP_NUM_THREADS"] = "10"
    os.environ["PYTHONPATH"] = "./"

    print(f"Starting Stage 1 Training...")
    print(f"Config  : {cfg_path}")
    print(f"Save to : {save_path}")
    print(f"GPUs    : {args.gpus}")
    if args.batch_size is not None:
        print(f"Batch   : {args.batch_size} (每卡 {args.batch_size // n_gpus})")
    if args.resume:
        print(f"Resume  : {args.resume}")
    if args.weight:
        print(f"Weight  : {args.weight}")

    try:
        train(cfg_path, opts)
    except Exception as e:
        print(f"Training interrupted: {e}")
        raise


if __name__ == "__main__":
    main()
