"""
Stage 1 VQ 重建质量指标，与 VOCA/FaceFormer/CodeTalker 论文使用的标准一致。

LVE (Lip Vertex Error, mm)
    每帧在 547 个嘴部顶点上取最大 L2 误差，再对所有帧取均值。
    越小越好，论文典型值：VOCA ~5.5mm, FaceFormer ~3.5mm, CodeTalker ~2.5mm。

MVE (Mean Vertex Error, mm)
    全脸所有顶点的平均 L2 误差（均值）。
    反映整体重建精度。
"""

import os
import numpy as np
import torch

_LVE_INDICES = None  # 延迟加载，避免每次 import 就读文件


def _load_lve_indices():
    global _LVE_INDICES
    if _LVE_INDICES is not None:
        return _LVE_INDICES
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "vocaset", "regions", "lve.txt")
    with open(path) as f:
        idx = [int(x.strip()) for x in f.read().split(",") if x.strip()]
    _LVE_INDICES = torch.tensor(idx, dtype=torch.long)
    return _LVE_INDICES


def compute_lve_mve(pred, gt):
    """
    计算单条序列的 LVE 和 MVE。

    参数
    ----
    pred : Tensor [T, V, 3]  预测顶点（绝对坐标，单位 m）
    gt   : Tensor [T, V, 3]  真实顶点（绝对坐标，单位 m）

    返回
    ----
    lve_mm : float  Lip Vertex Error (mm)
    mve_mm : float  Mean Vertex Error (mm)
    """
    lve_idx = _load_lve_indices().to(pred.device)

    # LVE: 每帧嘴部顶点最大 L2，再对帧取均值
    dist_lip = torch.norm(pred[:, lve_idx] - gt[:, lve_idx], p=2, dim=-1)  # [T, N_lip]
    lve_mm = dist_lip.max(dim=-1).values.mean().item() * 1000.0

    # MVE: 全脸顶点 L2 均值
    dist_all = torch.norm(pred - gt, p=2, dim=-1)  # [T, V]
    mve_mm = dist_all.mean().item() * 1000.0

    return lve_mm, mve_mm


class MetricMeter:
    """对多条序列的指标做累积平均。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self._lve_sum = 0.0
        self._mve_sum = 0.0
        self._count = 0

    def update(self, pred_flat, gt_flat, n_verts=5023):
        """
        pred_flat / gt_flat : Tensor [T, V*3]  （模型输出，单位 m）
        """
        T = pred_flat.shape[0]
        pred = pred_flat.view(T, n_verts, 3)
        gt   = gt_flat.view(T, n_verts, 3)
        lve, mve = compute_lve_mve(pred, gt)
        self._lve_sum += lve
        self._mve_sum += mve
        self._count += 1

    @property
    def lve(self):
        return self._lve_sum / self._count if self._count else 0.0

    @property
    def mve(self):
        return self._mve_sum / self._count if self._count else 0.0
