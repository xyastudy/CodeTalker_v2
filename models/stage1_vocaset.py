import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from models.lib.base_models import LinearEmbedding, PositionalEncoding, Transformer
from models.lib.quantizer import VectorQuantizer
from utils.indices_util import get_region_indices


class VQAutoEncoder(BaseModel):
    """ VQ-GAN model """
    def __init__(self, args):
        super().__init__()
        self.encoder = TransformerEncoder(args)
        # 三个独立 decoder，各自只接收来自本区域 loss 的梯度，消除梯度冲突
        self.decoder_lip = ConformerDecoder(args, args.in_dim)
        self.decoder_eye = ConformerDecoder(args, args.in_dim)
        self.decoder_other = ConformerDecoder(args, args.in_dim)

        self.quantize1 = VectorQuantizer(args.n_embed, args.zquant_dim, beta=0.25)
        self.quantize2 = VectorQuantizer(args.n_embed, args.zquant_dim, beta=0.25)
        self.quantize3 = VectorQuantizer(args.n_embed, args.zquant_dim, beta=0.25)

        self.proj1 = nn.Linear(1024, 1024)
        self.proj2 = nn.Linear(1024, 1024)
        self.proj3 = nn.Linear(1024, 1024)

        self.args = args

        region_indices = get_region_indices()
        if region_indices is not None:
            lip_region, eye_region, other_region = region_indices
            
            def create_mask(indices):
                mask = torch.zeros(15069)
                coords = torch.cat([indices * 3, indices * 3 + 1, indices * 3 + 2])
                mask[coords.long()] = 1.0
                return mask.view(1, 1, -1)

            self.register_buffer('mask_lip', create_mask(lip_region))
            self.register_buffer('mask_eye', create_mask(eye_region))
            self.register_buffer('mask_other', create_mask(other_region))
        else:
            print("Warning: region_indices is None")

    def encode(self, x, x_a=None):
        """ 使用三码本分区域量化 """
        h = self.encoder(x)  # h=torch.Size([B, 帧数, 1024])
        B, L, C = h.shape

        # 每个分支先做线性变换，再 view 成量化器需要的 h1, h2, h3 = [B, 帧数*16, 64]
        h1 = self.proj1(h).view(B, -1, self.args.zquant_dim) 
        h2 = self.proj2(h).view(B, -1, self.args.zquant_dim) 
        h3 = self.proj3(h).view(B, -1, self.args.zquant_dim)
        # print(f"DEBUG:h1.shape={h1.shape}, h2.shape={h2.shape}, h3.shape={h3.shape}") 
        
        # 分别量化 quant1, quant2, quant3 = [B, 64, 帧数*16]
        quant1, loss1, info1 = self.quantize1(h1)
        quant2, loss2, info2 = self.quantize2(h2)
        quant3, loss3, info3 = self.quantize3(h3)
        # print(f"DEBUG:quant1.shape={quant1.shape}, quant2.shape={quant2.shape}, quant3.shape={quant3.shape}")

        quant_full = torch.cat([quant1, quant2, quant3], dim=1)
        
        emb_loss = loss1 + loss2 + loss3
        
        # 索引记录
        indices = torch.stack([info1[2], info2[2], info3[2]], dim=-1) # [B, Lq, 3]      
        perplexity = (info1[0] + info2[0] + info3[0]) / 3.0
        info = (perplexity, None, indices)

        return quant_full, emb_loss, info  

    def decode(self, quant, decoder):
        #BCL
        quant = quant.permute(0,2,1)
        quant = quant.view(quant.shape[0], -1, self.args.face_quan_num, self.args.zquant_dim).contiguous()
        quant = quant.view(quant.shape[0], -1,  self.args.face_quan_num*self.args.zquant_dim).contiguous()
        quant = quant.permute(0,2,1).contiguous()
        dec = decoder(quant)
        return dec

    def forward(self, x, template):
        template = template.unsqueeze(1) # [B, 1, V*3]
        x_offset = x - template

        quant_full, emb_loss, info = self.encode(x_offset)  # quant_full=[B, 192, Lq]  

        quant_lip = quant_full[:, :64, :]
        quant_eye = quant_full[:, 64:128, :]
        quant_other = quant_full[:, 128:, :]

        offset_lip = self.decode(quant_lip, self.decoder_lip)
        offset_eye = self.decode(quant_eye, self.decoder_eye)
        offset_other = self.decode(quant_other, self.decoder_other)

        loss_punish_lip = torch.mean((offset_lip * (1 - self.mask_lip))**2)
        loss_punish_eye = torch.mean((offset_eye * (1 - self.mask_eye))**2)
        loss_punish_other = torch.mean((offset_other * (1 - self.mask_other))**2)
        loss_zero = loss_punish_lip + loss_punish_eye + loss_punish_other

        offset_lip = offset_lip * self.mask_lip    
        offset_eye = offset_eye * self.mask_eye    
        offset_other = offset_other * self.mask_other  

        full_vertices = offset_lip + offset_eye + offset_other + template

        lip_vertices = offset_lip + template
        eye_vertices = offset_eye + template
        other_vertices = offset_other + template

        info = (info[0], (lip_vertices, eye_vertices, other_vertices), info[2])  

        return full_vertices, emb_loss, info, loss_zero 

    def sample_step(self, x, x_a=None):
        quant_z, _, info = self.encode(x, x_a)  # quant_z: [B, 192, Lq]
        B, C, Lq = quant_z.shape

        q1 = quant_z[:, :64, :]
        q2 = quant_z[:, 64:128, :]
        q3 = quant_z[:, 128:, :]
        off1 = self.decode(q1, self.decoder_lip)
        off2 = self.decode(q2, self.decoder_eye)
        off3 = self.decode(q3, self.decoder_other)
        x_sample_det = off1 + off2 + off3

        btc = (B, Lq, C)
        indices = info[2]
        x_sample_check = self.decode_to_img(indices, btc)
        return x_sample_det, x_sample_check

    def get_quant(self, x, x_a=None):
        quant_z, _, info = self.encode(x, x_a)
        indices = info[2]
        return quant_z, indices

    def get_distances(self, x):
        h = self.encoder(x) ## x --> z'
        d = self.quantize.get_distance(h)
        return d

    def get_quant_from_d(self, d, btc):
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        x = self.decode_to_img(min_encoding_indices, btc)
        return x

    @torch.no_grad()
    def entry_to_feature(self, index, zshape):
        """
        实现索引拼接
        index: (B, Lq, 3)
        zshape: (B, Lq, C) 
        """
        index = index.long()
        B, Lq, C = zshape
        
        # 分别从三个码本取出 Embedding (B*Lq, split_size)
        e1 = self.quantize1.get_codebook_entry(index[:, :, 0].reshape(-1), shape=None)
        e2 = self.quantize2.get_codebook_entry(index[:, :, 1].reshape(-1), shape=None)
        e3 = self.quantize3.get_codebook_entry(index[:, :, 2].reshape(-1), shape=None)
        
        # 调整形状并拼接
        e1 = e1.view(B, Lq, -1)
        e2 = e2.view(B, Lq, -1)
        e3 = e3.view(B, Lq, -1)
        
        quant_z = torch.cat([e1, e2, e3], dim=-1) # (B, Lq, C)
        return quant_z

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        """
        index: (B*Lq, 1, 3) or (B, Lq, 3)
        zshape: (B, Lq, C)
        """
        quant_blc = self.entry_to_feature(index, zshape)  # (B, Lq, 192)
        q1 = quant_blc[:, :, :64].permute(0, 2, 1).contiguous()
        q2 = quant_blc[:, :, 64:128].permute(0, 2, 1).contiguous()
        q3 = quant_blc[:, :, 128:].permute(0, 2, 1).contiguous()
        off1 = self.decode(q1, self.decoder_lip)
        off2 = self.decode(q2, self.decoder_eye)
        off3 = self.decode(q3, self.decoder_other)
        return off1 + off2 + off3

    @torch.no_grad()
    def decode_logit(self, logits, zshape):
        if logits.dim() == 3:
            probs = F.softmax(logits, dim=-1)
            _, ix = torch.topk(probs, k=1, dim=-1)
        else:
            ix = logits
        ix = torch.reshape(ix, (-1,1))
        x = self.decode_to_img(ix, zshape)
        return x

    def get_logit(self, logits, sample=True, filter_value=-float('Inf'),
                  temperature=0.7, top_p=0.9, sample_idx=None):
        """ function that samples the distribution of logits. (used in test)
        if sample_idx is None, we perform nucleus sampling
        """
        logits = logits / temperature
        sample_idx = 0
        ##########
        probs = F.softmax(logits, dim=-1) # B, N, embed_num
        if sample:
            ## multinomial sampling
            shape = probs.shape
            probs = probs.reshape(shape[0]*shape[1],shape[2])
            ix = torch.multinomial(probs, num_samples=sample_idx+1)
            probs = probs.reshape(shape[0],shape[1],shape[2])
            ix = ix.reshape(shape[0],shape[1])
        else:
            ## top 1; no sampling
            _, ix = torch.topk(probs, k=1, dim=-1)
        return ix, probs


class TransformerEncoder(nn.Module):
  """ 编码器：将3D顶点数据映射到隐空间 """
  def __init__(self, args):
    super().__init__()
    self.args = args
    size = self.args.in_dim
    dim = self.args.hidden_size
    self.vertice_mapping = nn.Sequential(nn.Linear(size,dim), nn.LeakyReLU(self.args.neg, True))
    if args.quant_factor == 0:
        layers = [nn.Sequential(
                    nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                    )]
    else:
        layers = [nn.Sequential(
                    nn.Conv1d(dim,dim,5,stride=2,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                    )]
        for _ in range(1, args.quant_factor):
            layers += [nn.Sequential(
                        nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                    padding_mode='replicate'),
                        nn.LeakyReLU(self.args.neg, True),
                        nn.InstanceNorm1d(dim, affine=args.INaffine),
                        nn.MaxPool1d(2)
                        )]
    self.squasher = nn.Sequential(*layers)
    self.encoder_transformer = Transformer(
        in_size=self.args.hidden_size,
        hidden_size=self.args.hidden_size,
        num_hidden_layers=\
                self.args.num_hidden_layers,
        num_attention_heads=\
                self.args.num_attention_heads,
        intermediate_size=\
                self.args.intermediate_size)
    self.encoder_pos_embedding = PositionalEncoding(
        self.args.hidden_size)
    self.encoder_linear_embedding = LinearEmbedding(
        self.args.hidden_size,
        self.args.hidden_size)

  def forward(self, inputs):
    ## downdample into path-wise length seq before passing into transformer
    dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    inputs = self.vertice_mapping(inputs)
    inputs = self.squasher(inputs.permute(0,2,1)).permute(0,2,1) # [N L C]

    encoder_features = self.encoder_linear_embedding(inputs)
    encoder_features = self.encoder_pos_embedding(encoder_features)
    encoder_features = self.encoder_transformer((encoder_features, dummy_mask))

    return encoder_features


class TransformerDecoder(nn.Module):
  """ 解码器：将量化后的离散编码还原成3D顶点 """
  def __init__(self, args, out_dim, is_audio=False):
    super().__init__()
    self.args = args
    size=self.args.hidden_size
    dim=self.args.hidden_size
    self.expander = nn.ModuleList()
    if args.quant_factor == 0:
        self.expander.append(nn.Sequential(
                    nn.Conv1d(size,dim,5,stride=1,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                            ))
    else:
        self.expander.append(nn.Sequential(
                    nn.ConvTranspose1d(size,dim,5,stride=2,padding=2,
                                        output_padding=1,
                                        padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                            ))                      
        num_layers = args.quant_factor+2 \
            if is_audio else args.quant_factor

        for _ in range(1, num_layers):
            self.expander.append(nn.Sequential(
                                nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                        padding_mode='replicate'),
                                nn.LeakyReLU(self.args.neg, True),
                                nn.InstanceNorm1d(dim, affine=args.INaffine),
                                ))
    self.decoder_transformer = Transformer(
        in_size=self.args.hidden_size,
        hidden_size=self.args.hidden_size,
        num_hidden_layers=\
            self.args.num_hidden_layers,
        num_attention_heads=\
            self.args.num_attention_heads,
        intermediate_size=\
            self.args.intermediate_size)
    self.decoder_pos_embedding = PositionalEncoding(
        self.args.hidden_size)
    self.decoder_linear_embedding = LinearEmbedding(
        self.args.hidden_size,
        self.args.hidden_size)

    self.vertice_map_reverse = nn.Linear(args.hidden_size,out_dim)

    self.part_embedding = nn.Embedding(3, self.args.hidden_size) # 3个区域，每个维度 1024

  def forward(self, inputs, part_id):
    dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    ## upsample into original length seq before passing into transformer
    for i, module in enumerate(self.expander):
        inputs = module(inputs)
        if i > 0:
            inputs = inputs.repeat_interleave(2, dim=2)
    inputs = inputs.permute(0,2,1) #BLC
    part_emb = self.part_embedding(part_id).unsqueeze(1)
    decoder_features = self.decoder_linear_embedding(inputs)
    decoder_features = decoder_features + part_emb
    decoder_features = self.decoder_pos_embedding(decoder_features)

    decoder_features = self.decoder_transformer((decoder_features, dummy_mask))
    pred_recon = self.vertice_map_reverse(decoder_features)
    return pred_recon


class ConvModule(nn.Module):
    """Conformer 卷积模块：局部时序建模，对帧间连续的面部运动建模更友好。
    用 GroupNorm 代替 BatchNorm，兼容 batch_size=1。
    """
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * 2)      # 逐点扩展，用于 GLU 门控
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size,
                                  padding=kernel_size // 2,
                                  groups=dim,
                                  padding_mode='replicate')
        self.gn = nn.GroupNorm(min(32, dim // 16), dim)  # 代替 BN，batch_size=1 安全
        self.act = nn.SiLU()
        self.pw2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):   # x: [B, L, dim]
        residual = x
        x = self.norm(x)
        x = self.pw1(x)                          # [B, L, dim*2]
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * torch.sigmoid(x2)              # GLU: [B, L, dim]
        x = x.transpose(1, 2)                   # [B, dim, L]
        x = self.dw_conv(x)
        x = self.gn(x)
        x = x.transpose(1, 2)                   # [B, L, dim]
        x = self.act(x)
        x = self.pw2(x)
        x = self.dropout(x)
        return residual + x


class ConformerBlock(nn.Module):
    """Conformer Block: Macaron-FF → MHSA → ConvModule → Macaron-FF → LN。
    每个 FF 以半步残差 (×0.5) 形式加入，与原始 Conformer 论文一致。
    """
    def __init__(self, dim, num_heads, intermediate_size, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ff1_norm = nn.LayerNorm(dim)
        self.ff1 = nn.Sequential(
            nn.Linear(dim, intermediate_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, dim),
            nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConvModule(dim, kernel_size, dropout)
        self.ff2_norm = nn.LayerNorm(dim)
        self.ff2 = nn.Sequential(
            nn.Linear(dim, intermediate_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, dim),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):   # x: [B, L, dim]
        x = x + 0.5 * self.ff1(self.ff1_norm(x))
        normed = self.attn_norm(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.attn_drop(attn_out)
        x = self.conv(x)
        x = x + 0.5 * self.ff2(self.ff2_norm(x))
        return self.final_norm(x)


class ConformerDecoder(nn.Module):
    """Conformer 解码器：替换 TransformerDecoder，提供更强的局部+全局建模能力。

    相比原版 TransformerDecoder 的改进：
    - 每层加入 ConvModule，捕捉局部帧间连续性（对面部动画重建至关重要）
    - intermediate_size 独立配置（decoder_intermediate_size），默认 2048 vs 原来的 1536
    - 层数独立配置（decoder_num_layers），默认与编码器相同
    - 使用 SiLU 激活（比 GELU 在重建任务上更稳定）
    - 正弦位置编码正确按序列长度索引（修复了原版按 batch 维度索引的问题）
    """
    def __init__(self, args, out_dim):
        super().__init__()
        self.args = args
        dim = args.hidden_size

        # ——— 上采样模块（与 TransformerDecoder 保持一致，兼容 quant_factor） ———
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

        # ——— 正弦位置编码（按序列长度 L 正确索引） ———
        max_len = 5000
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)  # [max_len, dim]

        # ——— Conformer 主体 ———
        num_layers = getattr(args, 'decoder_num_layers', args.num_hidden_layers)
        intermediate = getattr(args, 'decoder_intermediate_size', args.intermediate_size)
        kernel_size = getattr(args, 'conv_kernel_size', 31)

        self.blocks = nn.ModuleList([
            ConformerBlock(dim, args.num_attention_heads, intermediate, kernel_size)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, out_dim)

    def forward(self, inputs):   # inputs: [B, dim, L]
        for i, module in enumerate(self.expander):
            inputs = module(inputs)
            if i > 0:
                inputs = inputs.repeat_interleave(2, dim=2)

        x = inputs.permute(0, 2, 1)                          # [B, L, dim]
        x = x + self.pe[:x.size(1), :].unsqueeze(0)          # 位置编码（按 L 索引）

        for block in self.blocks:
            x = block(x)

        return self.out_proj(self.out_norm(x))                # [B, L, out_dim]