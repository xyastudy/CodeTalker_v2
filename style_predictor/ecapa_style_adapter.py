import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
from speechbrain.inference import EncoderClassifier


# 必须导入 ProjHead 结构，确保与训练时一致
class ProjHead(nn.Module):
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


class EcapaStyleAdapter:
    def __init__(self, gallery_path, ckpt_path, out_dim, device="cuda", temperature=0.2):
        self.device = device
        self.temperature = temperature
        self.out_dim = out_dim

        print(f"[Adapter] Loading Gallery from {gallery_path}...")
        # Gallery 存储的是 {spk_id: normalized_centroid_vector}
        raw_gallery = np.load(gallery_path, allow_pickle=True).item()
        
        # 将 Gallery 转换为 Tensor 矩阵以便快速计算 [Num_Speakers, Embedding_Dim]
        # 注意：这里假设 key 是字符串形式的数字索引，我们按整数排序以对应 one-hot 顺序
        self.sorted_spk_ids = sorted(raw_gallery.keys(), key=lambda x: int(x))
        gallery_list = [torch.from_numpy(raw_gallery[sid]) for sid in self.sorted_spk_ids]
        self.gallery_matrix = torch.stack(gallery_list).to(device).float()

        # 加载 ProjHead 权重
        print(f"[Adapter] Loading ProjHead from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        emb_dim = checkpoint.get("emb_dim", 192) # 默认 ECAPA 维度
        self.proj = ProjHead(emb_dim).to(device)
        self.proj.load_state_dict(checkpoint["proj_state_dict"])
        self.proj.eval()

        # 加载冻结的 ECAPA 骨干
        print("[Adapter] Loading ECAPA backbone...")
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="./pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        self.ecapa.eval()

    @torch.no_grad()
    def predict_mixed_one_hot_from_wav_path(self, wav_path, sample_rate=16000, num_views=10, crop_len=32000, topk=5):
        # 1. 加载音频
        audio, sr = librosa.load(wav_path, sr=sample_rate)
        audio_t = torch.from_numpy(audio).unsqueeze(0).to(self.device).float() # (1, L)

        # 2. 多视图裁剪与特征提取 (与训练逻辑对齐)
        # 模拟训练时的 multi_view_crop
        L = audio_t.size(1)
        views = []
        for _ in range(num_views):
            if L > crop_len:
                start = torch.randint(0, L - crop_len + 1, (1,), device=self.device).item()
                views.append(audio_t[:, start : start + crop_len])
            else:
                views.append(audio_t)
        
        batch_views = torch.cat(views, dim=0) # (V, crop_len)
        
        # ECAPA 编码
        emb = self.ecapa.encode_batch(batch_views)
        if emb.dim() == 3:
            emb = emb.squeeze(1)
        
        # 归一化后取平均，再归一化 (稳定特征)
        emb = F.normalize(emb, dim=1).mean(dim=0, keepdim=True)
        emb = F.normalize(emb, dim=1)

        # 3. 关键：通过 ProjHead 投影到去内容化的空间
        z = self.proj(emb) # (1, D)

        # 4. 计算余弦相似度
        # z: (1, D), gallery_matrix: (S, D)
        similarities = torch.matmul(z, self.gallery_matrix.t()) # (1, S)
        
        # 5. Softmax 转换为混合概率 (Mixed One-Hot)
        logits = similarities / self.temperature
        probs = F.softmax(logits, dim=-1) # (1, S)

        # 6. 整理返回信息
        top_probs, top_indices = torch.topk(probs[0], k=min(topk, len(self.sorted_spk_ids)))
        
        info = {
            "top1_id": self.sorted_spk_ids[top_indices[0].item()],
            "top1_prob": top_probs[0].item(),
            "confidence": (top_probs[0] - top_probs[1]).item() if len(top_probs) > 1 else top_probs[0].item(),
            "topk": [(self.sorted_spk_ids[top_indices[i].item()], top_probs[i].item()) for i in range(len(top_probs))]
        }

        # 如果主模型的 one-hot 维度与 Gallery 人数不一致（通常是一致的），这里需要做映射
        # 这里默认返回 (1, S) 的概率分布作为混合标签
        return probs, info