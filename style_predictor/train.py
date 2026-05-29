import os
import sys
import re
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from transformers import Wav2Vec2Model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print("Added PROJECT_ROOT to sys.path:", PROJECT_ROOT)

from base.utilities import get_parser
from dataset.data_loader import get_dataloaders
from model_predictor import StyleEmbedder, AAMSoftmax


# -------------------------
# Config
# -------------------------
total_epochs = 200
CFG_PATH = "/home/an/code/CodeTalker/StylePredictor/config/predictor.yaml"
SAVE_DIR = "/home/an/code/CodeTalker/StylePredictor/checkpoint"

# audio config
SAMPLE_RATE = 16000
CROP_SECONDS = 2.0
CROP_LEN = int(SAMPLE_RATE * CROP_SECONDS)  # 32000


# -------------------------
# Utils: speaker id from filename
# -------------------------
_speaker_re = re.compile(r"FaceTalk_(\d{6}_\d{5})_")


def extract_speaker_id(file_name: str) -> str:
    """
    FaceTalk_170904_00128_TA_sentence14.wav -> 170904_00128
    """
    m = _speaker_re.search(file_name)
    if m:
        return m.group(1)
    m2 = re.search(r"(\d{6}_\d{5})", file_name)
    return m2.group(1) if m2 else file_name


def print_speaker_stats(items_by_spk, prefix="[VAL]", max_print=10):
    print(f"{prefix} speakers: {len(items_by_spk)}")
    for i, (spk, items) in enumerate(items_by_spk.items()):
        if i >= max_print:
            print(f"  ... ({len(items_by_spk) - max_print} more speakers)")
            break
        print(f"  - {spk}: {len(items)} utterances")


# -------------------------
# Augmentations
# -------------------------
def random_crop_1d(x, crop_len):
    # x: (B, L)
    B, L = x.shape
    if L <= crop_len:
        return x
    start = torch.randint(0, L - crop_len + 1, (B,), device=x.device)
    out = []
    for b in range(B):
        out.append(x[b, start[b] : start[b] + crop_len])
    return torch.stack(out, dim=0)


def add_noise(x, snr_db_low=15, snr_db_high=30):
    # x: (B, L)
    snr_db = torch.empty((x.size(0), 1), device=x.device).uniform_(snr_db_low, snr_db_high)
    sig_power = x.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-8)
    snr = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr
    noise = torch.randn_like(x) * noise_power.sqrt()
    return x + noise


def random_gain(x, gain_db_low=-6.0, gain_db_high=6.0):
    # x: (B,L)
    g = torch.empty((x.size(0), 1), device=x.device).uniform_(gain_db_low, gain_db_high)
    scale = (10.0 ** (g / 20.0))
    return x * scale


def make_two_views(audio, crop_len=CROP_LEN):
    """
    Make two identity-preserving views from the same cropped segment:
    crop -> (noise+gain) twice
    """
    crop = random_crop_1d(audio, crop_len)
    v1 = random_gain(add_noise(crop, 15, 30))
    v2 = random_gain(add_noise(crop, 15, 30))
    return v1, v2


def cos_dist(z1, z2):
    # z1,z2: normalized (B,D)
    return (1.0 - (z1 * z2).sum(dim=1)).mean()

# -------------------------
# Evaluation: embedding intra-speaker compactness
# -------------------------
@torch.no_grad()
def evaluate_embed_consistency(loader, embedder, wav2vec, device, prefix="[VAL]"):
    embedder.eval()
    wav2vec.eval()

    zs_by_spk = defaultdict(list)

    for batch in loader:
        if len(batch) == 5:
            audio, _, _, _, file_name = batch
        elif len(batch) == 4:
            audio, _, _, file_name = batch
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")

        audio = audio.to(device)  # (B,L)

        # deterministic-ish: use a fixed crop by taking the first CROP_LEN if possible
        if audio.size(1) > CROP_LEN:
            audio_eval = audio[:, :CROP_LEN]
        else:
            audio_eval = audio

        h = wav2vec(audio_eval).last_hidden_state
        z = embedder(h).detach().cpu().numpy()  # (B,D)

        fn_list = [str(x) for x in file_name] if isinstance(file_name, (list, tuple)) else [str(file_name)] * len(z)
        for fn, zi in zip(fn_list, z):
            spk = extract_speaker_id(fn)
            zs_by_spk[spk].append(zi)

    intra = []
    for spk, arr in zs_by_spk.items():
        if len(arr) < 2:
            continue
        A = np.stack(arr, 0)
        c = A.mean(0, keepdims=True)
        d = np.linalg.norm(A - c, axis=1).mean()
        intra.append(d)

    print_speaker_stats(zs_by_spk, prefix=prefix)

    return {
        "num_speakers": len(zs_by_spk),
        "mean_intra_dist": float(np.mean(intra)) if intra else 0.0,
    }


def train():
    sys.argv = [sys.argv[0], "--config", CFG_PATH]
    cfg = get_parser()
    cfg.read_audio = True

    subjects_list = cfg.train_subjects.split()
    S = len(subjects_list)
    print(f"[INFO] train_subjects={S}")

    loaders = get_dataloaders(cfg)
    train_loader = loaders["train"]
    valid_loader = loaders["valid"]
    test_loader = loaders["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)
    print(f"[INFO] crop_len={CROP_LEN} samples ({CROP_SECONDS:.1f}s @ {SAMPLE_RATE}Hz)")

    # wav2vec frozen
    wav2vec = Wav2Vec2Model.from_pretrained(cfg.wav2vec2model_path).to(device)
    wav2vec.eval()
    for p in wav2vec.parameters():
        p.requires_grad = False

    # embedder + AAM head
    embed_dim = 192
    embedder = StyleEmbedder(in_dim=wav2vec.config.hidden_size, emb_dim=256, out_dim=embed_dim).to(device)
    aam = AAMSoftmax(emb_dim=embed_dim, num_classes=S, margin=0.2, scale=30.0).to(device)

    optimizer = optim.Adam(list(embedder.parameters()) + list(aam.parameters()), lr=1e-4)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_path = os.path.join(SAVE_DIR, f"style_embedder_aam_{S}sub_best.pth")
    final_path = os.path.join(SAVE_DIR, f"style_embedder_aam_{S}sub_final.pth")

    best_val_intra = float("inf")

    for epoch in range(total_epochs):
        embedder.train()
        aam.train()

        steps = 0
        total_loss = 0.0
        cls_loss_m = 0.0
        cons_loss_m = 0.0
        correct = 0
        total_num = 0

        for step, batch in enumerate(train_loader):
            audio, _, _, one_hot, file_name = batch

            if epoch == 0 and step == 0:
                print("")
                print("==== Debug first batch ====")
                print("audio:", audio.shape, audio.dtype, audio.min().item(), audio.max().item())
                print("one_hot:", one_hot.shape, one_hot.dtype)
                print("file_name:", file_name)
                print("===========================")
                print("")

            audio = audio.to(device)  # (B,L)
            label = torch.argmax(one_hot.to(device), dim=1)  # (B,)

            # (A) supervised AAM classification (optionally crop to match training distribution)
            audio_sup = random_crop_1d(audio, CROP_LEN) if audio.size(1) > CROP_LEN else audio
            with torch.no_grad():
                h = wav2vec(audio_sup).last_hidden_state
            z = embedder(h)
            loss_cls, logits = aam(z, label)

            # (B) two-view embedding consistency (same crop, different noise/gain)
            v1, v2 = make_two_views(audio, crop_len=CROP_LEN)
            with torch.no_grad():
                h1 = wav2vec(v1).last_hidden_state
                h2 = wav2vec(v2).last_hidden_state
            z1 = embedder(h1)
            z2 = embedder(h2)
            loss_cons = cos_dist(z1, z2)

            # total loss
            lam_cons = 0.1
            loss = loss_cls + lam_cons * loss_cons

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # stats
            pred = torch.argmax(logits, dim=1)
            correct += (pred == label).sum().item()
            total_num += label.numel()

            total_loss += loss.item()
            cls_loss_m += loss_cls.item()
            cons_loss_m += loss_cons.item()
            steps += 1

        acc = correct / max(total_num, 1)
        print(
            f"Epoch {epoch+1:03d} | "
            f"total {total_loss/max(steps,1):.4f} | "
            f"cls {cls_loss_m/max(steps,1):.4f} | "
            f"cons {cons_loss_m/max(steps,1):.4f} (lam={lam_cons}) | "
            f"train_acc {acc:.3f}"
        )

        if (epoch + 1) % 10 == 0:
            val_m = evaluate_embed_consistency(valid_loader, embedder, wav2vec, device, prefix="[VAL]")
            print(
                f"[VAL] Epoch {epoch+1:03d} | "
                f"speakers {val_m['num_speakers']} | "
                f"mean_intra_dist {val_m['mean_intra_dist']:.4f}"
            )

            # lower intra-dist is better
            if val_m["mean_intra_dist"] <= best_val_intra:
                best_val_intra = val_m["mean_intra_dist"]
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "embedder_state_dict": embedder.state_dict(),
                        "aam_state_dict": aam.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metric_mean_intra_dist": best_val_intra,
                        "train_acc": acc,
                        "embed_dim": embed_dim,
                        "crop_len": CROP_LEN,
                    },
                    best_path,
                )
                print(f"[CKPT] Saved best to {best_path} (best_mean_intra_dist={best_val_intra:.4f})")

    # save final
    torch.save(
        {
            "epoch": total_epochs,
            "embedder_state_dict": embedder.state_dict(),
            "aam_state_dict": aam.state_dict(),
            "best_metric_mean_intra_dist": best_val_intra,
            "embed_dim": embed_dim,
            "crop_len": CROP_LEN,
        },
        final_path,
    )
    print(f"[CKPT] Saved final to {final_path}")

    # test best
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        embedder.load_state_dict(ckpt["embedder_state_dict"])
        print(f"[TEST] Loaded best checkpoint: {best_path} (best_mean_intra_dist={ckpt.get('best_metric_mean_intra_dist')})")

    test_m = evaluate_embed_consistency(test_loader, embedder, wav2vec, device, prefix="[TEST]")
    print(
        f"[TEST] speakers {test_m['num_speakers']} | "
        f"mean_intra_dist {test_m['mean_intra_dist']:.4f}"
    )

    # test final
    if os.path.isfile(final_path):
        ckpt = torch.load(final_path, map_location=device)
        embedder.load_state_dict(ckpt["embedder_state_dict"])
        print(f"[TEST] Loaded final checkpoint: {final_path} (best_mean_intra_dist={ckpt.get('best_metric_mean_intra_dist')})")

    test_m = evaluate_embed_consistency(test_loader, embedder, wav2vec, device, prefix="[TEST]")
    print(
        f"[TEST] speakers {test_m['num_speakers']} | "
        f"mean_intra_dist {test_m['mean_intra_dist']:.4f}"
    )


if __name__ == "__main__":
    train()
