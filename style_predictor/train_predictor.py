import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import Wav2Vec2Model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print("Added PROJECT_ROOT to sys.path:", PROJECT_ROOT)

from base.utilities import get_parser
from style_predictor import StylePredictor
from dataset.data_loader import get_dataloaders
import re
from collections import defaultdict, Counter


# -------------------------
# Config
# -------------------------
total_epochs = 200
CFG_PATH = "/home/an/code/CodeTalker/StylePredictor/config/predictor.yaml"
SAVE_DIR = "/home/an/code/CodeTalker/StylePredictor/checkpoint"



_speaker_re = re.compile(r"FaceTalk_(\d{6}_\d{5})_")


def extract_speaker_id(file_name: str) -> str:
    """
    FaceTalk_170904_00128_TA_sentence14.wav -> 170904_00128
    """
    m = _speaker_re.search(file_name)
    if m:
        return m.group(1)
    # 如果格式变了，尽量抓一个 6位_5位
    m2 = re.search(r"(\d{6}_\d{5})", file_name)
    return m2.group(1) if m2 else file_name


def print_speaker_stats(preds_by_spk, prefix="[VAL]", max_print=10):
    print(f"{prefix} speakers: {len(preds_by_spk)}")

    for i, (spk, preds) in enumerate(preds_by_spk.items()):
        if i >= max_print:
            print(f"  ... ({len(preds_by_spk) - max_print} more speakers)")
            break
        print(f"  - {spk}: {len(preds)} utterances")


def random_crop_1d(x, crop_len):
    # x: (B, L)
    B, L = x.shape
    if L <= crop_len:
        return x
    start = torch.randint(0, L - crop_len + 1, (B,), device=x.device)
    out = []
    for b in range(B):
        out.append(x[b, start[b]:start[b]+crop_len])
    return torch.stack(out, dim=0)


def add_noise(x, snr_db_low=10, snr_db_high=30):
    # x: (B, L)
    # 随机 SNR，加高斯噪声
    snr_db = torch.empty((x.size(0), 1), device=x.device).uniform_(snr_db_low, snr_db_high)
    sig_power = x.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-8)
    snr = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr
    noise = torch.randn_like(x) * noise_power.sqrt()
    return x + noise


def js_divergence(p, q, eps=1e-8):
    # p,q: (B, S) 概率分布
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * ( (p * (p.log() - m.log())).sum(dim=1)
                 + (q * (q.log() - m.log())).sum(dim=1) ).mean()


def rampup(epoch, ramp_epochs=30):
    if epoch < 0: return 0.0
    if epoch >= ramp_epochs: return 1.0
    x = epoch / ramp_epochs
    return float(x * x)  


@torch.no_grad()
def evaluate_consistency(loader, predictor, wav2vec, device, prefix="[VAL]"):
    predictor.eval()
    wav2vec.eval()

    preds_by_spk = defaultdict(list)  

    for batch in loader:
        if len(batch) == 5:
            audio, vertice, template, one_hot, file_name = batch
        elif len(batch) == 4:
            audio, vertice, template, file_name = batch
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")

        audio = audio.to(device)

        h = wav2vec(audio).last_hidden_state
        logits = predictor(h)
        pred = torch.argmax(logits, dim=1).detach().cpu().tolist()

        if isinstance(file_name, (list, tuple)):
            fn_list = [str(x) for x in file_name]
        else:
            fn_list = [str(file_name)] * len(pred)

        assert len(fn_list) == len(pred), f"file_name count != batch size: {len(fn_list)} vs {len(pred)}"

        for fn, p in zip(fn_list, pred):
            spk = extract_speaker_id(fn)
            preds_by_spk[spk].append(p)

    mode_cons_list = []
    all_equal_list = []

    for spk, preds in preds_by_spk.items():
        if len(preds) <= 1:
            continue
        mode_pred = Counter(preds).most_common(1)[0][0]
        mode_cons = sum(int(x == mode_pred) for x in preds) / len(preds)
        mode_cons_list.append(mode_cons)

        all_equal_list.append(1.0 if len(set(preds)) == 1 else 0.0)

    print_speaker_stats(preds_by_spk, prefix=prefix)

    return {
        "num_speakers": len(preds_by_spk),
        "mode_consistency": float(np.mean(mode_cons_list)) if mode_cons_list else 0.0,
        "all_equal_rate": float(np.mean(all_equal_list)) if all_equal_list else 0.0,
    }


def train():
    sys.argv = [sys.argv[0], "--config", CFG_PATH]
    cfg = get_parser()
    cfg.read_audio = True

    subjects_list = cfg.train_subjects.split()
    S = len(subjects_list)

    loaders = get_dataloaders(cfg)
    train_loader = loaders["train"]
    valid_loader = loaders["valid"]
    test_loader = loaders["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = StylePredictor(num_classes=S).to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    wav2vec = Wav2Vec2Model.from_pretrained(cfg.wav2vec2model_path).to(device)
    wav2vec.eval()
    for p in wav2vec.parameters():
        p.requires_grad = False

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_path = os.path.join(SAVE_DIR, f"style_predictor_{S}sub_best.pth")
    best_val_cons = -1.0

    for epoch in range(total_epochs):
        predictor.train()

        steps = 0
        total_loss = 0.0
        ce_loss = 0.0
        cons_loss = 0.0
        correct = 0
        
        for step, batch in enumerate(train_loader):
            audio, vertice, template, one_hot, file_name = batch

            if epoch == 0 and step == 0:
                print("")
                print("==== Debug first batch ====")
                print("audio:", audio.shape, audio.dtype, audio.min().item(), audio.max().item())
                print("vertice:", vertice.shape, vertice.dtype)
                print("template:", template.shape, template.dtype)
                print("one_hot:", one_hot.shape, one_hot.dtype)
                print("file_name:", file_name)
                print("===========================")
                print("")

            audio = audio.to(device)      # (B, L)
            one_hot = one_hot.to(device)  # (B, S)
            label = torch.argmax(one_hot, dim=1)  # (B,)

            crop_len = 32000  # 例如 2秒 @16k，按你数据采样率调整

            a_teacher = audio  
            a_student = random_crop_1d(audio, crop_len)
            a_student = add_noise(a_student)

            with torch.no_grad():
                h_t = wav2vec(a_teacher).last_hidden_state
                h_s = wav2vec(a_student).last_hidden_state  

            logits_t = predictor(h_t)     # teacher logits
            logits_s = predictor(h_s)     # student logits

            ce = criterion(logits_t, label)

            p_t = torch.softmax(logits_t.detach(), dim=1)
            p_s = torch.softmax(logits_s, dim=1)

            cons = js_divergence(p_s, p_t)   

            lam_max = 1
            lam = lam_max * rampup(epoch, ramp_epochs=100)
            loss = ce + lam * cons

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(logits_t, dim=1)
            correct += (pred == label).sum().item()
            total_loss += loss.item()
            ce_loss += ce.item()
            cons_loss += cons.item()
            steps += 1

        avg_total = total_loss / max(steps, 1)
        avg_ce = ce_loss / max(steps, 1)
        avg_cons = cons_loss / max(steps, 1)
        acc = correct / max(steps, 1)

        ratio = (lam * avg_cons) / (avg_ce + 1e-8)

        print(
            f"Epoch {epoch+1:03d} | "
            f"total_loss {avg_total:.4f} | ce_loss {avg_ce:.4f} | cons_loss {avg_cons:.4f} | "
            f"acc {acc:.3f} "
            f"cons/ce contribution ≈ {ratio:.3f}"
        )

        if (epoch + 1) % 10 == 0:
            val_m = evaluate_consistency(valid_loader, predictor, wav2vec, device)
            print(
                f"[VAL] Epoch {epoch+1:03d} | "
                f"speakers {val_m['num_speakers']} | "
                f"mode_cons {val_m['mode_consistency']:.3f} | "
                f"all_equal {val_m['all_equal_rate']:.3f}"
            )

            # 用 mode_consistency 作为选 best 的指标
            if val_m["mode_consistency"] >= best_val_cons:
                best_val_cons = val_m["mode_consistency"]
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "predictor_state_dict": predictor.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metric": best_val_cons,
                        "acc_val": acc,
                    },
                    best_path,
                )
                print(f"[CKPT] Saved best to {best_path} (best_mode_cons={best_val_cons:.3f},acc_val={acc})")

    final_path = os.path.join(SAVE_DIR, f"style_predictor_{S}sub_final.pth")
    torch.save(
        {
            "epoch": total_epochs,
            "predictor_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_val_cons,
        },
        final_path,
    )
    print(f"[CKPT] Saved final to {final_path}")

    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        predictor.load_state_dict(ckpt["predictor_state_dict"])
        print(f"[TEST] Loaded best checkpoint: {best_path} (best_mode_cons={ckpt.get('best_metric')},acc={ckpt.get('acc_val')})")

    test_m = evaluate_consistency(test_loader, predictor, wav2vec, device, prefix="[TEST]")
    print(
        f"[TEST] speakers {test_m['num_speakers']} | "
        f"mode_cons {test_m['mode_consistency']:.3f} | "
        f"all_equal {test_m['all_equal_rate']:.3f}"
    )

    if os.path.isfile(final_path):
        ckpt = torch.load(final_path, map_location=device)
        predictor.load_state_dict(ckpt["predictor_state_dict"])
        print(f"[TEST] Loaded final checkpoint: {final_path} (best_mode_cons={ckpt.get('best_metric')},acc={ckpt.get('acc_val')})")

    test_m = evaluate_consistency(test_loader, predictor, wav2vec, device, prefix="[TEST]")
    print(
        f"[TEST] speakers {test_m['num_speakers']} | "
        f"mode_cons {test_m['mode_consistency']:.3f} | "
        f"all_equal {test_m['all_equal_rate']:.3f}"
    )


if __name__ == "__main__":
    train()
