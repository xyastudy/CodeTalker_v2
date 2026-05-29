"""
python tools/lip_sync_metrics.py
"""

import os
import sys
import glob

import numpy as np
import torch
from tqdm import tqdm
from scipy.interpolate import interp1d

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from base.utilities import get_logger
from utils.indices_util import get_region_indices

# ================= config =================
mask_file = "vocaset/regions/FLAME_masks.pkl" 
gt_dir = "vocaset/vertices_npy"
# 单码本模型结果文件夹
baseline_dir = "/home/an/code/CodeTalker/RUN/vocaset/CodeTalker_s2/result/npy"
# 三码本模型结果文件夹
proposed_dir = "RUN/vocaset/CodeTalker_s2/result/npy/"
# ===========================================

mask_file = os.path.join(project_root, mask_file)
gt_dir = os.path.join(project_root, gt_dir)
baseline_dir = os.path.join(project_root, baseline_dir)
proposed_dir = os.path.join(project_root, proposed_dir)


def align_and_evaluate(pred_dir, gt_dir, lip_indices):
    """ 遍历文件夹并计算指标 """
    mve_all, lve_all, max_lve_all = [], [], []
    
    pred_files = glob.glob(os.path.join(pred_dir, "*.npy"))
    if not pred_files:
        return None

    for pred_path in tqdm(pred_files, desc=f"Evaluating {os.path.basename(pred_dir)}", leave=False):
        filename = os.path.basename(pred_path)
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

            f_pred, f_gt = pred.shape[0], gt.shape[0]
            if f_pred != f_gt:
                x_old = np.linspace(0, 1, f_pred)
                x_new = np.linspace(0, 1, f_gt)
                f_interp = interp1d(x_old, pred.reshape(f_pred, -1), axis=0, kind='linear', fill_value="extrapolate")
                pred_aligned = f_interp(x_new).reshape(f_gt, 5023, 3)
            else:
                pred_aligned = pred

            pred_t = torch.from_numpy(pred_aligned).float()
            gt_t = torch.from_numpy(gt).float()
            dist_full = torch.norm(pred_t - gt_t, p=2, dim=-1)
            mve_all.append(torch.mean(dist_full).item())
            dist_lips = torch.norm(pred_t[:, lip_indices, :] - gt_t[:, lip_indices, :], p=2, dim=-1)
            lve_all.append(torch.mean(dist_lips).item())
            max_lve_all.append(torch.max(dist_lips).item())

        except Exception as e:
            print(f"Error in {filename}: {e}")
            continue

    return {
        "mve": np.mean(mve_all) * 1000, 
        "lve": np.mean(lve_all) * 1000, 
        "max_lve": np.mean(max_lve_all) * 1000
    }

def main():
    lip_indices = get_region_indices()[0]

    global logger
    logger = get_logger()
    
    logger.info("开始对比实验评估...")
    base_res = align_and_evaluate(baseline_dir, gt_dir, lip_indices)
    prop_res = align_and_evaluate(proposed_dir, gt_dir, lip_indices)

    if base_res and prop_res:

        for key in ["mve", "lve", "max_lve"]:
            b, p = base_res[key], prop_res[key]
            imp = (b - p) / b * 100
            name = key
            logger.info(f"    {name:<10} | baseline:{b:>10.4f} | proposed:{p:>10.4f} | improvement:{imp:>10.2f}%")
        

if __name__ == "__main__":
    main()