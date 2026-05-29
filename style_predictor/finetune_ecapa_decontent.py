#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Useage:
    python StylePredictor/finetune_ecapa_decontent.py
"""

import os
import sys

import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from collections import defaultdict
from typing import Dict, Any, Tuple
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
from speechbrain.inference import EncoderClassifier

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from base import config
from dataset.data_loader import get_dataloaders

# ========================== Config ==========================
cfg_path = os.path.join(project_root, "style_predictor/config/predictor.yaml")
save_dir = os.path.join(project_root, "style_predictor/checkpoint")

sample_rate = 16000
crop_len = 32000          
num_views = 5             
temperature = 0.2
epoch = 500
lr = 1e-3                 
weight_decay = 1e-4
lam_dc = 1.0
neg_margin = 0.2

_sentence_re = re.compile(r"sentence(\d+)")


def extract_sentence_id(fn: str) -> int:
    m = _sentence_re.search(fn)
    return int(m.group(1)) if m else -1


def multi_view_crop(audio_1xL: torch.Tensor, crop_len: int, num_views: int) -> torch.Tensor:
    """audio_1xL: (1,L) -> (V, crop_len_or_L)"""
    device = audio_1xL.device
    L = audio_1xL.size(1)
    views = []
    for _ in range(num_views):
        if L > crop_len:
            start = torch.randint(0, L - crop_len + 1, (1,), device=device).item()
            views.append(audio_1xL[:, start:start + crop_len])
        else:
            views.append(audio_1xL)
    return torch.cat(views, dim=0)


def decontent_style_loss(
    z: torch.Tensor,      # (B,D) normalized
    spk: torch.Tensor,    # (B,)
    sent: torch.Tensor,   # (B,)
    neg_margin: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Positive pairs: same spk, different sent  => push cos -> 1
      loss_pos = mean(1 - cos)
    Negative pairs: different spk, same sent  => push cos <= neg_margin
      loss_neg = mean(relu(cos - neg_margin))
    """
    B, D = z.shape
    sim = z @ z.t()  # (B,B)

    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    pos_mask = (spk[:, None] == spk[None, :]) & (sent[:, None] != sent[None, :]) & (~eye)
    neg_mask = (spk[:, None] != spk[None, :]) & (sent[:, None] == sent[None, :]) & (~eye)

    pos_pairs = int(pos_mask.sum().item())
    neg_pairs = int(neg_mask.sum().item())

    loss_pos = torch.tensor(0.0, device=z.device)
    loss_neg = torch.tensor(0.0, device=z.device)

    if pos_pairs > 0:
        loss_pos = (1.0 - sim[pos_mask]).mean()

    if neg_pairs > 0:
        loss_neg = F.relu(sim[neg_mask] - neg_margin).mean()

    loss = loss_pos + loss_neg
    stats = {
        "pos_pairs": pos_pairs,
        "neg_pairs": neg_pairs,
        "loss_pos": float(loss_pos.detach().cpu()),
        "loss_neg": float(loss_neg.detach().cpu()),
    }
    return loss, stats


class ProjHead(nn.Module):
    """Tiny projection head: D -> D (with LayerNorm)"""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, dim=1)


@torch.no_grad()
def build_gallery(train_loader, ecapa, proj, device, crop_len=crop_len, num_views=num_views) -> Dict[str, np.ndarray]:
    """Build centroid gallery using ECAPA->proj embeddings."""
    z_by_spk = defaultdict(list)

    for batch in train_loader:
        audio = batch[0].to(device)           # (B,L)
        one_hot = batch[5].to(device)         # (B,S)
        label = torch.argmax(one_hot, dim=1)  # (B,)

        B = audio.size(0)
        for b in range(B):
            views = multi_view_crop(audio[b:b+1], crop_len=crop_len, num_views=num_views)  # (V, crop)
            emb = ecapa.encode_batch(views)  # (V,1,D) or (V,D)
            if emb.dim() == 3:
                emb = emb.squeeze(1)
            emb = F.normalize(emb, dim=1)
            emb = emb.mean(dim=0, keepdim=True)              # (1,D)
            emb = F.normalize(emb, dim=1)
            z = proj(emb)                                    # (1,D) normalized
            z_by_spk[str(int(label[b].item()))].append(z.squeeze(0).cpu().numpy())

    gallery = {}
    for spk, arr in z_by_spk.items():
        A = np.stack(arr, 0)
        c = A.mean(0)
        c = c / (np.linalg.norm(c) + 1e-12)
        gallery[spk] = c
    return gallery


@torch.no_grad()
def unsupervised_validation(valid_loader, ecapa, proj, device, args, n_clusters):
    """
    无监督评估：
    n_clusters: 预设的聚类数量。可以设为你训练集中说话人的大致数量。
    """
    proj.eval()
    all_embeddings = []

    # 1. 提取验证集所有嵌入向量
    for batch in valid_loader:
        audio = batch[0].to(device)  # (B, L)
        for b in range(audio.size(0)):
            # 使用多视图增强稳定性
            views = multi_view_crop(audio[b:b+1], crop_len=args.crop_len, num_views=args.num_views)
            emb = ecapa.encode_batch(views)
            if emb.dim() == 3:
                emb = emb.squeeze(1)
            emb = F.normalize(emb, dim=1).mean(dim=0, keepdim=True)
            emb = F.normalize(emb, dim=1)
            z = proj(emb).squeeze(0).cpu().numpy()
            all_embeddings.append(z)

    X = np.stack(all_embeddings, axis=0)  # (N, D)

    # 2. 执行 K-Means 聚类
    # 如果样本量太小，调低 n_clusters
    actual_clusters = min(n_clusters, len(X) // 2)
    kmeans = KMeans(n_clusters=actual_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # 3. 计算指标
    # Silhouette Score: 范围 [-1, 1]，越接近 1 代表聚类越完美
    sil_score = silhouette_score(X, cluster_labels)
    # Calinski-Harabasz: 值越大代表簇间距离越远，簇内越紧密
    ch_score = calinski_harabasz_score(X, cluster_labels)

    return {
        "silhouette": sil_score,
        "ch_index": ch_score,
        "embedding_std": np.std(X, axis=0).mean() # 监控特征是否坍缩（若std趋近0则说明模型失效）
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)
    print(f"[INFO] crop_len={crop_len} num_views={num_views}")

    cfg = config.load_cfg_from_cfg_file(cfg_path)
    cfg.read_audio = True
    subjects = cfg.train_subjects.split()
    S = len(subjects)
    print("[INFO] train_subjects:", S)

    loaders = get_dataloaders(cfg)
    train_loader = loaders["train"]
    valid_loader = loaders.get("valid", None)

    # Load ECAPA (frozen)
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(project_root, "pretrained_models", "spkrec-ecapa-voxceleb"),
        run_opts={"device": device},
    )
    ecapa.eval()
    for p in ecapa.parameters():
        p.requires_grad = False

    # Determine embedding dim by running one tiny forward
    with torch.no_grad():
        dummy = torch.zeros(1, 16000, device=device)  # 1 sec dummy
        emb = ecapa.encode_batch(dummy)
        if emb.dim() == 3:
            emb = emb.squeeze(1)
        D = emb.shape[-1]
    print("[INFO] ECAPA emb dim:", D)

    # Projection head (trainable)
    proj = ProjHead(D).to(device)

    opt = optim.AdamW(proj.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"ecapa_proj_decontent_S{S}.pth")
    gallery_path = os.path.join(save_dir, f"ecapa_gallery_proj_S{S}.npy")

    best_val = -1.0

    for epoch in range(1, epoch + 1):
        proj.train()

        loss_m = 0.0
        pos_pairs_m = 0
        neg_pairs_m = 0

        for batch in train_loader:
            audios, _, _, _, _, one_hots, names = batch
            audio = audios.to(device)           # (B,L)
            one_hot = one_hots.to(device)         # (B,S)
            file_name = names                  # list[str] or tuple[str]
            label = torch.argmax(one_hot, dim=1)  # (B,)

            if isinstance(file_name, (list, tuple)):
                fn_list = [str(x) for x in file_name]
            else:
                fn_list = [str(file_name)] * label.numel()

            sent_ids = torch.tensor([extract_sentence_id(fn) for fn in fn_list],
                                    device=device, dtype=torch.long)

            # Build embedding per sample with multi-view crop
            zs = []
            for b in range(audio.size(0)):
                views = multi_view_crop(audio[b:b+1], crop_len=crop_len, num_views=num_views)  # (V,crop)
                with torch.no_grad():
                    emb = ecapa.encode_batch(views)
                    if emb.dim() == 3:
                        emb = emb.squeeze(1)
                    emb = F.normalize(emb, dim=1).mean(dim=0, keepdim=True)  # (1,D)
                    emb = F.normalize(emb, dim=1)
                z = proj(emb)  # (1,D) normalized
                zs.append(z)
            z = torch.cat(zs, dim=0)  # (B,D)

            loss_dc, stats = decontent_style_loss(z, label, sent_ids, neg_margin=neg_margin)
            loss = lam_dc * loss_dc

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_m += float(loss.detach().cpu())
            pos_pairs_m += stats["pos_pairs"]
            neg_pairs_m += stats["neg_pairs"]

        loss_m /= max(len(train_loader), 1)

        print(f"Epoch {epoch:03d} | loss_dc {loss_m:.4f} (lam={lam_dc}) "
              f"| pos_pairs {pos_pairs_m} neg_pairs {neg_pairs_m}")

        # ===================== Train =====================
        if valid_loader is not None and (epoch % 10 == 0):
            metrics = unsupervised_validation(valid_loader, ecapa, proj, device, n_clusters=S)
            
            print(f"[VAL 无监督] Epoch {epoch}")
            print(f" >> 轮廓系数 (Silhouette): {metrics['silhouette']:.4f} (越接近1越好)")
            print(f" >> CH 指数: {metrics['ch_index']:.2f} (越大越好)")
            print(f" >> 特征标准差: {metrics['embedding_std']:.6f} (若过小可能发生坍缩)")

            # 使用轮廓系数作为保存最佳模型的标准
            if metrics['silhouette'] > best_val:
                best_val = metrics['silhouette']
                torch.save({"proj_state_dict": proj.state_dict(),
                            "emb_dim": D,
                            "train_subjects": subjects,
                            "best_val_mean_intra": best_val,
                            "temperature": temperature,
                            "crop_len": crop_len,
                            "num_views": num_views,
                            "neg_margin": neg_margin,
                            "lam_dc": lam_dc,}, ckpt_path, )
                print(f"[CKPT] saved best proj to: {ckpt_path}") 

    # Build & save gallery using trained proj
    print("[INFO] Building gallery with trained proj...")
    proj.eval()
    gallery = build_gallery(train_loader, ecapa, proj, device, crop_len=crop_len, num_views=num_views)
    np.save(gallery_path, gallery, allow_pickle=True)
    print(f"[DONE] saved gallery: {gallery_path} (size={len(gallery)})")
    print(f"[DONE] saved proj ckpt: {ckpt_path}")


if __name__ == "__main__":
    main()
