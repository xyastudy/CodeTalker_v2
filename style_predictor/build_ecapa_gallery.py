"""
Build ECAPA gallery from training set.

Usage:
  python style_predictor/build_ecapa_gallery.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from collections import defaultdict
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
out = os.path.join(project_root, "style_predictor/checkpoint/ecapa_gallery_8.npy")
num_views = 3
crop_len = 32000

cfg = config.load_cfg_from_cfg_file(cfg_path)


@torch.no_grad()
def ecapa_embed_views(ecapa, audio_1xL, num_views, crop_len):
    """
    audio_1xL: (1,L) tensor on device
    return: (D,) normalized embedding
    """
    device = audio_1xL.device
    L = audio_1xL.size(1)

    views = []
    for _ in range(num_views):
        if L > crop_len:
            start = torch.randint(0, L - crop_len + 1, (1,), device=device).item()
            views.append(audio_1xL[:, start:start + crop_len])
        else:
            views.append(audio_1xL)

    xb = torch.cat(views, dim=0)  # (V, crop)
    emb = ecapa.encode_batch(xb)  # (V,1,D) or (V,D)
    if emb.dim() == 3:
        emb = emb.squeeze(1)
    emb = F.normalize(emb, dim=1)
    emb = F.normalize(emb.mean(dim=0), dim=0)
    return emb


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)

    cfg.read_audio = True

    subjects = cfg.train_subjects.split()
    print(f"[INFO] train_subjects={len(subjects)}")

    loaders = get_dataloaders(cfg)
    train_loader = loaders["train"]

    # Load ECAPA
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(project_root, "style_predictor/pretrained_models/spkrec-ecapa-voxceleb"),
        run_opts={"device": device},
    )

    # Collect embeddings
    z_by_spk = defaultdict(list)

    for batch in train_loader:
        audio = batch[0].to(device)           # (B,L)
        one_hot = batch[3].to(device)         # (B,S)
        label = torch.argmax(one_hot, dim=1)  # (B,)

        B = audio.size(0)
        for b in range(B):
            z = ecapa_embed_views(
                ecapa,
                audio[b:b+1],
                num_views=num_views,
                crop_len=crop_len,
            )
            z_by_spk[str(label[b].item())].append(z.cpu().numpy())

    # Build centroids
    gallery = {}
    for spk, arr in z_by_spk.items():
        A = np.stack(arr, axis=0)   # (N,D)
        c = A.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        gallery[spk] = c
        print(f"[INFO] speaker {spk}: {len(arr)} utterances")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, gallery, allow_pickle=True)
    print(f"[DONE] Saved ECAPA gallery to {out}")
    print(f"[DONE] gallery size = {len(gallery)}")


if __name__ == "__main__":
    main()
