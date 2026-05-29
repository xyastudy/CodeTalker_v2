import os
import sys
import torch
from transformers import Wav2Vec2Model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from base.utilities import get_parser
from dataset.data_loader import get_dataloaders
from model_predictor import StyleEmbedder  
from inference import build_speaker_gallery, infer_style_from_audio

CFG_PATH = "/home/an/code/CodeTalker/StylePredictor/config/predictor.yaml"
CKPT_PATH = "/home/an/code/CodeTalker/StylePredictor/checkpoint/style_embedder_aam_8sub_best.pth"  # 改成你的best
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_models_and_gallery():
    # 1) cfg + loader
    sys.argv = [sys.argv[0], "--config", CFG_PATH]
    cfg = get_parser()
    cfg.read_audio = True

    loaders = get_dataloaders(cfg)
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    # 2) wav2vec
    wav2vec = Wav2Vec2Model.from_pretrained(cfg.wav2vec2model_path).to(DEVICE)
    wav2vec.eval()
    for p in wav2vec.parameters():
        p.requires_grad = False

    # 3) embedder + load ckpt
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    embed_dim = ckpt.get("embed_dim", 192)

    embedder = StyleEmbedder(in_dim=wav2vec.config.hidden_size, emb_dim=256, out_dim=embed_dim).to(DEVICE)
    embedder.load_state_dict(ckpt["embedder_state_dict"])
    embedder.eval()

    # 4) build gallery（用训练集标签）
    gallery = build_speaker_gallery(train_loader, embedder, wav2vec, DEVICE, use_label_from_onehot=True)
    print("[INFO] gallery size:", len(gallery))

    return cfg, train_loader, test_loader, wav2vec, embedder, gallery

def demo_infer_one_from_test():
    cfg, train_loader, test_loader, wav2vec, embedder, gallery = load_models_and_gallery()

    batch = next(iter(test_loader))
    # 兼容 test batch 可能是 4 或 5 项：音频永远是第 0 项，file_name 永远是最后一项
    audio = batch[0]              # (B,L)
    file_name = batch[-1]

    # 推理第一条
    wav = audio[0]
    fn = file_name[0] if isinstance(file_name, (list, tuple)) else file_name

    res = infer_style_from_audio(wav, embedder, wav2vec, gallery, DEVICE, topk=5, num_views=5)
    subjects_list = cfg.train_subjects.split()

    # res["top1"] 是 label_id 字符串
    top1_id = int(res["top1"])
    print("Query file:", fn)
    print("Top1 label:", res["top1"], "=> name:", subjects_list[top1_id], "score:", res["top1_score"], "conf:", res["confidence"])
    print("Top5:")
    for k, s in res["topk"]:
        print("  ", k, "=>", subjects_list[int(k)], "score:", s)

if __name__ == "__main__":
    demo_infer_one_from_test()
