"""
python tools/eval_geom.py

1、计算 预测顶点 与 真实顶点 的几何误差（MSE/MVE）
2、在计算过程中会处理预测帧数与真实帧数不一致时的“重采样对齐”问题
"""

import os
import sys

import numpy as np
import torch
import glob
from tqdm import tqdm
from scipy.interpolate import interp1d

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from base.utilities import get_logger

# 路径设置（相对路径）
pred_dir = "RUN/vocaset/CodeTalkerV2_s1/result/npy"
gt_dir = "vocaset/vertices_npy"

pred_dir = os.path.join(project_root, pred_dir)
gt_dir = os.path.join(project_root, gt_dir)


def resample_sequence(source_data, target_len):
    """ 将 source_data 从原始帧数重采样到 target_len """
    f_src = source_data.shape[0]
    if f_src == target_len:
        return source_data

    v_shape = source_data.shape[1:]
    flat_data = source_data.reshape(f_src, -1)
    
    x_old = np.linspace(0, 1, f_src)
    x_new = np.linspace(0, 1, target_len)
    
    f_interp = interp1d(x_old, flat_data, axis=0, kind='linear')
    resampled_flat = f_interp(x_new)
    
    return resampled_flat.reshape(target_len, *v_shape)


def calculate_batch_metrics_with_resample(pred_dir, gt_dir):
    mse_list = []
    mve_list = []

    global logger 
    logger = get_logger()

    pred_files = glob.glob(os.path.join(pred_dir, "*.npy"))
    logger.info(f"找到 {len(pred_files)} 个预测文件，开始进行重采样和评估...")

    for pred_path in tqdm(pred_files):
        filename = os.path.basename(pred_path)
        
        # 匹配预测顶点和真实顶点
        if "_condition_" not in filename:
            gt_filename = f"{filename}"
            gt_path = os.path.join(gt_dir, gt_filename)
        else:
            gt_filename = f"{filename.split('_condition_')[0]}.npy"
            gt_path = os.path.join(gt_dir, gt_filename)
        
        if not os.path.exists(gt_path):
            continue

        try:
            pred = np.load(pred_path).reshape(-1, 5023, 3)
            gt = np.load(gt_path).reshape(-1, 5023, 3)

            # 重采样：将预测结果 pred 对齐到 GT 的帧数
            pred_aligned = resample_sequence(pred, gt.shape[0])

            pred_t = torch.from_numpy(pred_aligned).float()
            gt_t = torch.from_numpy(gt).float()

            # MSE
            mse = torch.mean((pred_t - gt_t) ** 2).item()
            # MVE 
            mve = torch.mean(torch.norm(pred_t - gt_t, p=2, dim=-1)).item()

            mse_list.append(mse)
            mve_list.append(mve)
            
        except Exception as e:
            logger.info(f"Error processing {filename}: {e}")
            continue

    if not mse_list:
        logger.info("未匹配到任何文件")
        return

    print("\n" + "="*50)
    print(f"平均 MSE: {np.mean(mse_list):.8f}")
    print(f"平均 MVE: {np.mean(mve_list)*1000:.4f} mm (重采样对齐后)")
    print("="*50)

if __name__ == "__main__":
    calculate_batch_metrics_with_resample(pred_dir, gt_dir)
