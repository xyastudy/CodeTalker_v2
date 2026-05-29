#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------
# 0) Make imports work (base/, dataset/ etc.)
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print("[INFO] Added PROJECT_ROOT:", PROJECT_ROOT)

# Your project imports
from base.utilities import get_parser
from dataset.data_loader import get_dataloaders


import torchaudio

# --- SpeechBrain compatibility patch for torchaudio >= 2.9 (backend APIs removed) ---
if not hasattr(torchaudio, "list_audio_backends"):
    def list_audio_backends():
        # SpeechBrain只需要它是个可迭代列表即可
        # 你这里用 soundfile 读wav也没关系
        return ["soundfile"]
    torchaudio.list_audio_backends = list_audio_backends
# -------------------------------------------------------------------------------


# ECAPA from SpeechBrain
# pip install speechbrain
from speechbrain.pretrained import EncoderClassifier

# Optional wav loader
try:
    import soundfile as sf  # pip install soundfile
    _HAS_SF = True
except Exception:
    _HAS_SF = False


# ------------------------------------------------------------
# Config defaults
# ------------------------------------------------------------
CFG_PATH_DEFAULT = "/home/an/code/CodeTalker/StylePredictor/config/predictor.yaml"
SAMPLE_RATE = 16000
CROP_SECONDS = 2.0
CROP_LEN = int(SAMPLE_RATE * CROP_SECONDS)


# ------------------------------------------------------------
# Audio helpers
# ------------------------------------------------------------
def load_wav_16k(path: str) -> torch.Tensor:
    """
    Load a mono wav and return float32 Tensor (L,) at 16k.
    Requires soundfile. If your wav is not 16k, you should resample outside
    or add resampling here.
    """
    if not _HAS_SF:
        raise RuntimeError("soundfile not installed. Run: pip install soundfile")
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)  # to mono
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"Expected {SAMPLE_RATE}Hz wav. Got {sr}Hz for {path}. "
                           f"Please resample to 16k before inference.")
    return torch.from_numpy(wav)


def multi_view_crop(audio: torch.Tensor, crop_len: int, num_views: int, device: str) -> torch.Tensor:
    """
    audio: (1,L) on device
    return: (num_views, crop_len) or (num_views, L) if L <= crop_len
    """
    assert audio.dim() == 2 and audio.size(0) == 1
    L = audio.size(1)
    views = []
    for _ in range(num_views):
        if L > crop_len:
            start = torch.randint(0, L - crop_len + 1, (1,), device=device).item()
            views.append(audio[:, start:start + crop_len])
        else:
            views.append(audio)
    return torch.cat(views, dim=0)  # (V, crop_len_or_L)


# ------------------------------------------------------------
# ECAPA embedding wrapper
# ------------------------------------------------------------
class EcapaEmbedder:
    def __init__(self, device: str):
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(PROJECT_ROOT, "pretrained_models", "spkrec-ecapa-voxceleb"),
            run_opts={"device": device},
        )

    @torch.no_grad()
    def embed_batch(self, audio_b: torch.Tensor) -> torch.Tensor:
        """
        audio_b: (B,L) float32 16k on device
        return: (B,D) normalized
        """
        emb = self.model.encode_batch(audio_b)  # (B,1,D) or (B,D) depending on version
        if emb.dim() == 3:
            emb = emb.squeeze(1)
        emb = F.normalize(emb, dim=1)
        return emb


# ------------------------------------------------------------
# Gallery (speaker centroid) build
# ------------------------------------------------------------
@torch.no_grad()
def build_gallery_from_train(
    train_loader,
    ecapa: EcapaEmbedder,
    device: str,
    crop_len: int = CROP_LEN,
    num_views: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Build gallery: speaker_label(str) -> centroid(D,)
    Assumes train batch contains one_hot at index 3:
      batch = (audio, _, _, one_hot, file_name)
    """
    z_by_spk = defaultdict(list)

    for batch in train_loader:
        # your train loader should be len==5
        if len(batch) < 4:
            raise ValueError("Unexpected batch format. Expected audio and one_hot in train_loader.")
        audio = batch[0].to(device)           # (B,L)
        one_hot = batch[3].to(device)         # (B,S)
        label = torch.argmax(one_hot, dim=1)  # (B,)

        # multi-view: crop several segments and average embeddings for stability
        B, L = audio.shape
        zs = []
        for b in range(B):
            a = audio[b:b+1]  # (1,L)
            views = multi_view_crop(a, crop_len=crop_len, num_views=num_views, device=device)  # (V, crop)
            z_v = ecapa.embed_batch(views)  # (V,D)
            z = F.normalize(z_v.mean(dim=0, keepdim=True), dim=1)  # (1,D)
            zs.append(z)
        z = torch.cat(zs, dim=0).detach().cpu().numpy()  # (B,D)

        for k, zi in zip(label.detach().cpu().tolist(), z):
            z_by_spk[str(k)].append(zi)

    gallery = {}
    for spk, arr in z_by_spk.items():
        A = np.stack(arr, 0)          # (N,D)
        c = A.mean(0)                 # (D,)
        c = c / (np.linalg.norm(c) + 1e-12)
        gallery[spk] = c

    return gallery


# ------------------------------------------------------------
# Single-utterance inference
# ------------------------------------------------------------
@torch.no_grad()
def infer_one(
    audio_1d: torch.Tensor,              # (L,) or (1,L)
    ecapa: EcapaEmbedder,
    gallery: Dict[str, np.ndarray],
    device: str,
    crop_len: int = CROP_LEN,
    num_views: int = 10,
    topk: int = 5,
) -> Dict:
    """
    Return:
      {
        "topk": [(label_str, score), ...],
        "top1": label_str,
        "top1_score": float,
        "confidence": top1 - top2,
        "embedding": np.ndarray(D,)
      }
    """
    if audio_1d.dim() == 1:
        audio = audio_1d.unsqueeze(0)
    else:
        audio = audio_1d
    audio = audio.to(device)  # (1,L)

    views = multi_view_crop(audio, crop_len=crop_len, num_views=num_views, device=device)  # (V, crop)
    z_v = ecapa.embed_batch(views)  # (V,D)
    z = F.normalize(z_v.mean(dim=0), dim=0)  # (D,)
    z_np = z.detach().cpu().numpy()

    scores = [(spk, float(np.dot(z_np, c))) for spk, c in gallery.items()]
    scores.sort(key=lambda x: x[1], reverse=True)

    return {
        "topk": scores[:topk],
        "top1": scores[0][0],
        "top1_score": scores[0][1],
        "confidence": (scores[0][1] - scores[1][1]) if len(scores) > 1 else 0.0,
        "embedding": z_np,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=CFG_PATH_DEFAULT)
    ap.add_argument("--wav", type=str, default="", help="Optional: path to a 16kHz mono wav for inference")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--num_views", type=int, default=10)
    ap.add_argument("--gallery_views", type=int, default=3, help="Num views per train utterance when building gallery")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)
    print(f"[INFO] crop_len={CROP_LEN} samples ({CROP_SECONDS:.1f}s @ {SAMPLE_RATE}Hz)")
    print("[INFO] num_views(infer):", args.num_views, " gallery_views:", args.gallery_views)

    # Load cfg using your parser
    sys.argv = [sys.argv[0], "--config", args.config]
    cfg = get_parser()
    cfg.read_audio = True
    subjects_list = cfg.train_subjects.split()
    print("[INFO] train_subjects:", len(subjects_list))

    # Load dataloaders
    loaders = get_dataloaders(cfg)
    train_loader = loaders["train"]
    test_loader = loaders.get("test", None)

    # Init ECAPA
    ecapa = EcapaEmbedder(device=device)

    # Build gallery
    gallery = build_gallery_from_train(
        train_loader=train_loader,
        ecapa=ecapa,
        device=device,
        crop_len=CROP_LEN,
        num_views=args.gallery_views,
    )
    print("[INFO] gallery size:", len(gallery))

    # Pick query audio
    if args.wav:
        wav = load_wav_16k(args.wav)
        query_name = args.wav
    else:
        if test_loader is None:
            raise RuntimeError("No test_loader found in get_dataloaders(cfg), and --wav not provided.")
        batch = next(iter(test_loader))
        audio = batch[0]     # (B,L)
        fname = batch[-1]    # list/tuple of names
        wav = audio[0].detach().cpu()
        query_name = fname[0] if isinstance(fname, (list, tuple)) else str(fname)

    # Inference
    res = infer_one(
        audio_1d=wav,
        ecapa=ecapa,
        gallery=gallery,
        device=device,
        crop_len=CROP_LEN,
        num_views=args.num_views,
        topk=args.topk,
    )

    # Pretty print
    def name_of(label_str: str) -> str:
        i = int(label_str)
        return subjects_list[i] if 0 <= i < len(subjects_list) else label_str

    print("Query:", query_name)
    print(f"Top1 label: {res['top1']} => name: {name_of(res['top1'])} "
          f"score: {res['top1_score']:.6f} conf: {res['confidence']:.6f}")
    print("TopK:")
    for k, s in res["topk"]:
        print(f"  {k} => {name_of(k)}  score: {s:.6f}")


if __name__ == "__main__":
    main()
