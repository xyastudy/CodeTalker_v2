"""
Route A: Single VQ-codebook autoencoder (baseline architecture).

与 git HEAD 原始 baseline 完全相同的模型结构。
区域加权损失由训练脚本 (train_vq_single.py) 外部计算，无需改动模型。
Stage 2 接口与原始 baseline 完全兼容。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from models.lib.base_models import LinearEmbedding, PositionalEncoding, Transformer
from models.lib.quantizer import VectorQuantizer


class VQAutoEncoder(BaseModel):
    def __init__(self, args):
        super().__init__()
        self.encoder = TransformerEncoder(args)
        self.decoder = TransformerDecoder(args, args.in_dim)
        self.quantize = VectorQuantizer(args.n_embed, args.zquant_dim, beta=0.25)
        self.args = args

    def encode(self, x, x_a=None):
        h = self.encoder(x)
        h = h.view(x.shape[0], -1, self.args.face_quan_num, self.args.zquant_dim)
        h = h.view(x.shape[0], -1, self.args.zquant_dim)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = quant.permute(0, 2, 1)
        quant = quant.view(quant.shape[0], -1, self.args.face_quan_num, self.args.zquant_dim).contiguous()
        quant = quant.view(quant.shape[0], -1, self.args.face_quan_num * self.args.zquant_dim).contiguous()
        quant = quant.permute(0, 2, 1).contiguous()
        return self.decoder(quant)

    def forward(self, x, template):
        template = template.unsqueeze(1)
        x_offset = x - template
        quant, emb_loss, info = self.encode(x_offset)
        dec = self.decode(quant)
        dec = dec + template
        return dec, emb_loss, info

    def sample_step(self, x, x_a=None):
        quant_z, _, info = self.encode(x, x_a)
        x_sample_det = self.decode(quant_z)
        btc = quant_z.shape[0], quant_z.shape[2], quant_z.shape[1]
        indices = info[2]
        x_sample_check = self.decode_to_img(indices, btc)
        return x_sample_det, x_sample_check

    def get_quant(self, x, x_a=None):
        quant_z, _, info = self.encode(x, x_a)
        indices = info[2]
        return quant_z, indices

    @torch.no_grad()
    def entry_to_feature(self, index, zshape):
        index = index.long()
        quant_z = self.quantize.get_codebook_entry(index.reshape(-1), shape=None)
        quant_z = torch.reshape(quant_z, zshape)
        return quant_z

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = index.long()
        quant_z = self.quantize.get_codebook_entry(index.reshape(-1), shape=None)
        quant_z = torch.reshape(quant_z, zshape).permute(0, 2, 1)
        return self.decode(quant_z)

    @torch.no_grad()
    def decode_logit(self, logits, zshape):
        if logits.dim() == 3:
            probs = F.softmax(logits, dim=-1)
            _, ix = torch.topk(probs, k=1, dim=-1)
        else:
            ix = logits
        ix = torch.reshape(ix, (-1, 1))
        return self.decode_to_img(ix, zshape)

    def get_logit(self, logits, sample=True, filter_value=-float('Inf'),
                  temperature=0.7, top_p=0.9, sample_idx=None):
        logits = logits / temperature
        sample_idx = 0
        probs = F.softmax(logits, dim=-1)
        if sample:
            shape = probs.shape
            probs = probs.reshape(shape[0] * shape[1], shape[2])
            ix = torch.multinomial(probs, num_samples=sample_idx + 1)
            probs = probs.reshape(shape[0], shape[1], shape[2])
            ix = ix.reshape(shape[0], shape[1])
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        return ix, probs


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        size = args.in_dim
        dim = args.hidden_size
        self.vertice_mapping = nn.Sequential(nn.Linear(size, dim), nn.LeakyReLU(args.neg, True))
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

    def forward(self, inputs):
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        inputs = self.vertice_mapping(inputs)
        inputs = self.squasher(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        encoder_features = self.encoder_linear_embedding(inputs)
        encoder_features = self.encoder_pos_embedding(encoder_features)
        encoder_features = self.encoder_transformer((encoder_features, dummy_mask))
        return encoder_features


class TransformerDecoder(nn.Module):
    def __init__(self, args, out_dim, is_audio=False):
        super().__init__()
        self.args = args
        size = args.hidden_size
        dim = args.hidden_size
        self.expander = nn.ModuleList()
        if args.quant_factor == 0:
            self.expander.append(nn.Sequential(
                nn.Conv1d(size, dim, 5, stride=1, padding=2, padding_mode='replicate'),
                nn.LeakyReLU(args.neg, True),
                nn.InstanceNorm1d(dim, affine=args.INaffine),
            ))
        else:
            self.expander.append(nn.Sequential(
                nn.ConvTranspose1d(size, dim, 5, stride=2, padding=2,
                                   output_padding=1, padding_mode='replicate'),
                nn.LeakyReLU(args.neg, True),
                nn.InstanceNorm1d(dim, affine=args.INaffine),
            ))
            num_layers = args.quant_factor + 2 if is_audio else args.quant_factor
            for _ in range(1, num_layers):
                self.expander.append(nn.Sequential(
                    nn.Conv1d(dim, dim, 5, stride=1, padding=2, padding_mode='replicate'),
                    nn.LeakyReLU(args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine),
                ))
        self.decoder_transformer = Transformer(
            in_size=args.hidden_size, hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size)
        self.decoder_pos_embedding = PositionalEncoding(args.hidden_size)
        self.decoder_linear_embedding = LinearEmbedding(args.hidden_size, args.hidden_size)
        self.vertice_map_reverse = nn.Linear(args.hidden_size, out_dim)

    def forward(self, inputs):
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        for i, module in enumerate(self.expander):
            inputs = module(inputs)
            if i > 0:
                inputs = inputs.repeat_interleave(2, dim=2)
        inputs = inputs.permute(0, 2, 1)
        decoder_features = self.decoder_linear_embedding(inputs)
        decoder_features = self.decoder_pos_embedding(decoder_features)
        decoder_features = self.decoder_transformer((decoder_features, dummy_mask))
        return self.vertice_map_reverse(decoder_features)
