"""
Route B 启动脚本：真三流 VQ 自编码器（三条流从输入顶点起完全隔离）。

用法:
  python run_vq_triplestream.py
  python run_vq_triplestream.py --exp_name my_triplestream_exp
  python run_vq_triplestream.py --gpus 0 1 --batch_size 2
  python run_vq_triplestream.py --resume RUN/vocaset/s1/xxx/model/train_epoch_50.pth
"""

import argparse
import os
import sys
from datetime import datetime

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main.train_vq_triplestream import main as train


def parse_args():
    parser = argparse.ArgumentParser(description="Route B: triple-stream VQ-AE training")
    parser.add_argument("--exp_name",   type=str, default=None)
    parser.add_argument("--resume",     type=str, default=None)
    parser.add_argument("--weight",     type=str, default=None)
    parser.add_argument("--gpus",       type=int, nargs="+", default=[0])
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    n_gpus = len(args.gpus)
    effective_bs = args.batch_size or 1
    if effective_bs < n_gpus:
        raise ValueError(f"batch_size ({effective_bs}) 必须 >= GPU 数量 ({n_gpus})")

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"triplestream_{now}"
    cfg_path  = os.path.join(project_root, "config/vocaset/stage1_triplestream.yaml")
    save_path = os.path.join(project_root, f"RUN/vocaset/s1/{exp_name}")

    opts = ["save_path", save_path, "train_gpu", str(args.gpus)]
    if args.batch_size:
        opts += ["batch_size", str(args.batch_size)]
    if args.resume:
        opts += ["resume", args.resume]
    if args.weight:
        opts += ["weight", args.weight]

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in args.gpus)
    os.environ["OMP_NUM_THREADS"] = "10"
    os.environ["PYTHONPATH"] = "./"

    print(f"[Route B] True triple-stream: lip / eye / other fully independent")
    print(f"Config  : {cfg_path}")
    print(f"Save to : {save_path}")
    print(f"GPUs    : {args.gpus}")

    train(cfg_path, opts)


if __name__ == "__main__":
    main()
