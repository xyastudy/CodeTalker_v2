"""
Route B: True triple-stream VQ autoencoder.

核心设计：三条流从输入就完全隔离。
  lip_region 顶点 → RegionEncoder → VQ1 → ConformerDecoder → lip 重建
  eye_region 顶点 → RegionEncoder → VQ2 → ConformerDecoder → eye 重建
  other 顶点     → RegionEncoder → VQ3 → ConformerDecoder → other 重建

每条流只看自己区域的顶点坐标，梯度不跨流干扰。
三码本的 perplexity 预期低于当前"伪三码本"（编码器输入真正不同）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from models.lib.base_models import LinearEmbedding, PositionalEncoding, Transformer
from models.lib.quantizer import VectorQuantizer
from utils.indices_util import get_region_indices


# ────────────────────────────── 编码器 ──────────────────────────────

class RegionEncoder(nn.Module):
    """TransformerEncoder with a configurable in_dim (not taken from args.in_dim)."""

    def __init__(self, args, in_dim):
        super().__init__()
        self.args = args
        dim = args.hidden_size
        self.vertice_mapping = nn.Sequential(
            nn.Linear(in_dim, dim), nn.LeakyReLU(args.neg, True)
        )
        if args.quant_factor == 0:
            layers = [nn.Sequential(
                nn.Conv1d(dim, dim, 5, stride=1, padding=2, padding_mode='replicate'),
                nn.LeakyReLU(args.neg, True),
                nn.InstanceNorm1d(dim, affine=args.INaffine),
            )]
        else:
            layers = [nn.Sequential(
                nn.Conv1d(dim, dim, 5, stride=2, padding=2, padding_mode='replicate'),
                nn.LeakyReLU(args.neg, True),
                nn.InstanceNorm1d(dim, affine=args.INaffine),
            )]
            for _ in range(1, args.quant_factor):
                layers += [nn.Sequential(
                    nn.Conv1d(dim, dim, 5, stride=1, padding=2, padding_mode='replicate'),
                    nn.LeakyReLU(args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine),
                    nn.MaxPool1d(2),
                )]
        self.squasher = nn.Sequential(*layers)
        self.encoder_transformer = Transformer(
            in_size=args.hidden_size, hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size)
        self.encoder_pos_embedding = PositionalEncoding(args.hidden_size)
        self.encoder_linear_embedding = LinearEmbedding(args.hidden_size, args.hidden_size)

    def forward(self, inputs):          # [B, T, in_dim] → [B, T', 1024]
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        inputs = self.vertice_mapping(inputs)
        inputs = self.squasher(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        features = self.encoder_linear_embedding(inputs)
        features = self.encoder_pos_embedding(features)
        return self.encoder_transformer((features, dummy_mask))


# ────────────────────────────── 解码器 ──────────────────────────────

class ConvModule(nn.Module):
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * 2)
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size,
                                  padding=kernel_size // 2, groups=dim,
                                  padding_mode='replicate')
        self.gn = nn.GroupNorm(min(32, dim // 16), dim)
        self.act = nn.SiLU()
        self.pw2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.pw1(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * torch.sigmoid(x2)
        x = x.transpose(1, 2)
        x = self.dw_conv(x)
        x = self.gn(x)
        x = x.transpose(1, 2)
        x = self.act(x)
        x = self.pw2(x)
        return residual + self.dropout(x)


class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads, intermediate_size, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ff1_norm = nn.LayerNorm(dim)
        self.ff1 = nn.Sequential(
            nn.Linear(dim, intermediate_size), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(intermediate_size, dim), nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConvModule(dim, kernel_size, dropout)
        self.ff2_norm = nn.LayerNorm(dim)
        self.ff2 = nn.Sequential(
            nn.Linear(dim, intermediate_size), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(intermediate_size, dim), nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ff1(self.ff1_norm(x))
        normed = self.attn_norm(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.attn_drop(attn_out)
        x = self.conv(x)
        x = x + 0.5 * self.ff2(self.ff2_norm(x))
        return self.final_norm(x)


class ConformerDecoder(nn.Module):
    """Conformer decoder with configurable out_dim (region-specific vertex count)."""

    def __init__(self, args, out_dim):
        super().__init__()
        self.args = args
        dim = args.hidden_size

        self.expander = nn.ModuleList()
        if args.quant_factor == 0:
            self.expander.append(nn.Sequential(
                nn.Conv1d(dim, dim, 5, stride=1, padding=2, padding_mode='replicate'),
                nn.LeakyReLU(args.neg, True),
                nn.InstanceNorm1d(dim, affine=args.INaffine),
            ))
        else:
            self.expander.append(nn.Sequential(
                nn.ConvTranspose1d(dim, dim, 5, stride=2, padding=2,
                                   output_padding=1, padding_mode='replicate'),
                nn.LeakyReLU(args.neg, True),
                nn.InstanceNorm1d(dim, affine=args.INaffine),
            ))
            for _ in range(1, args.quant_factor):
                self.expander.append(nn.Sequential(
                    nn.Conv1d(dim, dim, 5, stride=1, padding=2, padding_mode='replicate'),
                    nn.LeakyReLU(args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine),
                ))

        max_len = 5000
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

        num_layers = getattr(args, 'decoder_num_layers', args.num_hidden_layers)
        intermediate = getattr(args, 'decoder_intermediate_size', args.intermediate_size)
        kernel_size = getattr(args, 'conv_kernel_size', 31)
        self.blocks = nn.ModuleList([
            ConformerBlock(dim, args.num_attention_heads, intermediate, kernel_size)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, out_dim)

    def forward(self, inputs):          # [B, dim, L] → [B, L, out_dim]
        for i, module in enumerate(self.expander):
            inputs = module(inputs)
            if i > 0:
                inputs = inputs.repeat_interleave(2, dim=2)
        x = inputs.permute(0, 2, 1)
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(self.out_norm(x))


# ────────────────────────────── 主模型 ──────────────────────────────

class TripleStreamVQAE(BaseModel):
    """
    三条独立流的 VQ 自编码器。

    区域尺寸（5023 顶点 FLAME mesh）：
      lip:   250 verts → in_dim=750
      eye:   751 verts → in_dim=2253
      other: 4022 verts → in_dim=12066
    """

    TOTAL_VERTS = 5023

    def __init__(self, args):
        super().__init__()
        self.args = args

        lip_region, eye_region, other_region = get_region_indices()

        # 注册区域顶点索引（整型，不参与梯度）
        self.register_buffer('lip_region', lip_region)    # [N_lip]
        self.register_buffer('eye_region', eye_region)    # [N_eye]
        self.register_buffer('other_region', other_region)  # [N_other]

        # 预计算展平坐标索引 [v*3, v*3+1, v*3+2, ...] 用于在 [B,T,V*3] 中切片
        self.register_buffer('lip_coords',   self._coord_idx(lip_region))
        self.register_buffer('eye_coords',   self._coord_idx(eye_region))
        self.register_buffer('other_coords', self._coord_idx(other_region))

        lip_dim   = len(lip_region)   * 3   # 750
        eye_dim   = len(eye_region)   * 3   # 2253
        other_dim = len(other_region) * 3   # 12066

        # 三条独立编码器
        self.encoder_lip   = RegionEncoder(args, lip_dim)
        self.encoder_eye   = RegionEncoder(args, eye_dim)
        self.encoder_other = RegionEncoder(args, other_dim)

        # 三个独立码本
        self.quantize_lip   = VectorQuantizer(args.n_embed, args.zquant_dim, beta=0.25)
        self.quantize_eye   = VectorQuantizer(args.n_embed, args.zquant_dim, beta=0.25)
        self.quantize_other = VectorQuantizer(args.n_embed, args.zquant_dim, beta=0.25)

        # 三条独立解码器（out_dim 各不同）
        self.decoder_lip   = ConformerDecoder(args, lip_dim)
        self.decoder_eye   = ConformerDecoder(args, eye_dim)
        self.decoder_other = ConformerDecoder(args, other_dim)

    @staticmethod
    def _coord_idx(region):
        """vertex indices → flattened XYZ coord indices, e.g. [v0*3, v0*3+1, v0*3+2, ...]"""
        return torch.stack([region * 3, region * 3 + 1, region * 3 + 2], dim=1).reshape(-1)

    # ── 内部 encode/decode 辅助 ──────────────────────────────────────

    def _quant(self, h, quantizer):
        """h: [B, T, 1024] → quant (BCL), emb_loss, info"""
        B = h.shape[0]
        h = h.view(B, -1, self.args.zquant_dim)       # [B, T*16, 64]
        return quantizer(h)                             # quant: [B, 64, T*16]

    def _decode(self, quant, decoder):
        """quant: [B, 64, T*16] BCL → [B, T, out_dim]"""
        q = quant.permute(0, 2, 1).contiguous()        # [B, T*16, 64]
        B = q.shape[0]
        q = q.view(B, -1, self.args.face_quan_num, self.args.zquant_dim)   # [B, T, 16, 64]
        q = q.view(B, -1, self.args.face_quan_num * self.args.zquant_dim)  # [B, T, 1024]
        q = q.permute(0, 2, 1).contiguous()            # [B, 1024, T]
        return decoder(q)                               # [B, T, out_dim]

    # ── 公共接口 ─────────────────────────────────────────────────────

    def encode(self, x_offset):
        """
        x_offset: [B, T, V*3]  (x - template, template 已在外部减去)
        返回: quants=(q_lip, q_eye, q_other), emb_loss, info
        info = (perplexity, None, indices:[N,1,3])
        """
        x_lip   = x_offset[:, :, self.lip_coords]
        x_eye   = x_offset[:, :, self.eye_coords]
        x_other = x_offset[:, :, self.other_coords]

        h_lip   = self.encoder_lip(x_lip)
        h_eye   = self.encoder_eye(x_eye)
        h_other = self.encoder_other(x_other)

        q_lip,   l_lip,   i_lip   = self._quant(h_lip,   self.quantize_lip)
        q_eye,   l_eye,   i_eye   = self._quant(h_eye,   self.quantize_eye)
        q_other, l_other, i_other = self._quant(h_other, self.quantize_other)

        emb_loss = l_lip + l_eye + l_other
        perplexity = (i_lip[0] + i_eye[0] + i_other[0]) / 3.0
        indices = torch.stack([i_lip[2], i_eye[2], i_other[2]], dim=-1)  # [N, 1, 3]
        info = (perplexity, None, indices)

        return (q_lip, q_eye, q_other), emb_loss, info

    def forward(self, x, template):
        """
        返回:
          full_vertices: [B, T, V*3]  完整重建（含 template）
          emb_loss:      标量
          info:          (perplexity, (dec_lip, dec_eye, dec_other), indices)
                         dec_* 是各区域的 offset 预测，供训练脚本计算区域损失
        """
        template = template.unsqueeze(1)          # [B, 1, V*3]
        x_offset = x - template                   # [B, T, V*3]

        (q_lip, q_eye, q_other), emb_loss, info = self.encode(x_offset)

        dec_lip   = self._decode(q_lip,   self.decoder_lip)    # [B, T, 750]
        dec_eye   = self._decode(q_eye,   self.decoder_eye)    # [B, T, 2253]
        dec_other = self._decode(q_other, self.decoder_other)  # [B, T, 12066]

        # 拼装完整 mesh
        B, T = x.shape[:2]
        full_offset = torch.zeros(B, T, self.TOTAL_VERTS * 3, device=x.device)
        full_offset[:, :, self.lip_coords]   = dec_lip
        full_offset[:, :, self.eye_coords]   = dec_eye
        full_offset[:, :, self.other_coords] = dec_other

        full_vertices = full_offset + template

        # 把区域 offset 放进 info[1]，供训练脚本计算区域损失
        info = (info[0], (dec_lip, dec_eye, dec_other), info[2])
        return full_vertices, emb_loss, info

    def get_quant(self, x_offset):
        """Stage 2 兼容接口（预留）。"""
        (q_lip, q_eye, q_other), _, info = self.encode(x_offset)
        indices = info[2]
        # 把三路 quant 沿 channel 拼接，保持 BCL 格式
        quant_full = torch.cat([q_lip, q_eye, q_other], dim=1)  # [B, 192, Lq]
        return quant_full, indices

    @torch.no_grad()
    def decode_to_img(self, indices, zshape):
        """
        从三路码本索引重建完整 mesh（Stage 2 / test 用）。
        indices: [N, 1, 3]  N = B * Lq
        zshape:  (B, Lq, 64)
        """
        B, Lq, _ = zshape
        idx_lip   = indices[:, :, 0].reshape(-1)
        idx_eye   = indices[:, :, 1].reshape(-1)
        idx_other = indices[:, :, 2].reshape(-1)

        e_lip   = self.quantize_lip.get_codebook_entry(idx_lip,   shape=None).view(B, Lq, -1)
        e_eye   = self.quantize_eye.get_codebook_entry(idx_eye,   shape=None).view(B, Lq, -1)
        e_other = self.quantize_other.get_codebook_entry(idx_other, shape=None).view(B, Lq, -1)

        # BLC → BCL
        q_lip   = e_lip.permute(0, 2, 1).contiguous()
        q_eye   = e_eye.permute(0, 2, 1).contiguous()
        q_other = e_other.permute(0, 2, 1).contiguous()

        dec_lip   = self._decode(q_lip,   self.decoder_lip)
        dec_eye   = self._decode(q_eye,   self.decoder_eye)
        dec_other = self._decode(q_other, self.decoder_other)

        T = dec_lip.shape[1]
        full = torch.zeros(B, T, self.TOTAL_VERTS * 3, device=dec_lip.device)
        full[:, :, self.lip_coords]   = dec_lip
        full[:, :, self.eye_coords]   = dec_eye
        full[:, :, self.other_coords] = dec_other
        return full
