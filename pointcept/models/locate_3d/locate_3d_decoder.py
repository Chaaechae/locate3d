"""
Locate3D transformer decoder.

Adapted from facebookresearch/locate-3d (models/locate_3d_decoder.py) so that
the decoder can be plugged on top of an arbitrary 3D encoder, supports variable
length batched point clouds via key-padding masks, and returns intermediate
layer outputs for deep supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .bbox_utils import box_cxcyczwhd_to_xyzxyz_jit


class LearnedPosEmbeddings(nn.Module):
    def __init__(self, dim=3, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(dim, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1),
        )

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        pos = self.position_embedding_head(xyz)
        return pos.transpose(1, 2)


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout, drop_path, is_self_attn):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.query_norm = nn.LayerNorm(d_model)
        self.keys_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.is_self_attn = is_self_attn
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries, queries_pos_embed, keys, keys_pos_embed, mask):
        if self.is_self_attn:
            normed_queries = self.query_norm(queries)
            normed_keys = self.query_norm(keys)
        else:
            normed_queries = self.query_norm(queries)
            normed_keys = keys

        q = (normed_queries + queries_pos_embed).transpose(0, 1)
        k = (normed_keys + keys_pos_embed).transpose(0, 1)
        attn_out = self.attn(q, k, k, key_padding_mask=mask, need_weights=False)[0]
        attn_out = attn_out.transpose(0, 1)

        queries = queries + self.drop_path(self.dropout(attn_out))
        queries = queries + self.drop_path(self.ffn(self.ffn_norm(queries)))
        return queries


class TransformerModule(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout, drop_path, use_checkpointing):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.query_self_attn = Block(d_model, n_heads, dim_feedforward, dropout, drop_path, is_self_attn=True)
        self.query_ptc_feat_attn = Block(d_model, n_heads, dim_feedforward, dropout, drop_path, is_self_attn=False)
        self.ptc_feat_query_attn = Block(d_model, n_heads, dim_feedforward, dropout, drop_path, is_self_attn=False)

    def _run(self, fn, *args):
        if self.use_checkpointing and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, query_feats, query_pos_embed, text_feats, text_pos_embed,
                ptc_feats, ptc_pos_embed, query_mask, ptc_mask):
        joint_feats = torch.cat([query_feats, text_feats], dim=1)
        joint_pos = torch.cat([query_pos_embed, text_pos_embed], dim=1)

        joint_feats = self._run(
            self.query_self_attn, joint_feats, joint_pos, joint_feats, joint_pos, query_mask,
        )
        joint_feats = self._run(
            self.query_ptc_feat_attn, joint_feats, joint_pos, ptc_feats, ptc_pos_embed, ptc_mask,
        )
        ptc_feats = self._run(
            self.ptc_feat_query_attn, ptc_feats, ptc_pos_embed, joint_feats, joint_pos, query_mask,
        )

        n_text = text_feats.shape[1]
        text_feats = joint_feats[:, -n_text:]
        query_feats = joint_feats[:, :-n_text]
        return query_feats, text_feats, ptc_feats


class MaskPredictionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mask_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, query_feats, ptc_feats):
        query_feats = self.mask_embed(query_feats)
        return torch.einsum("bqc,bnc->bqn", query_feats, ptc_feats)


class TextAlignmentHead(nn.Module):
    def __init__(self, d_model, max_tokens):
        super().__init__()
        self.text_alignment_head = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(d_model, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(d_model, max_tokens, 1),
        )

    def forward(self, query_feats):
        scores = self.text_alignment_head(query_feats.permute(0, 2, 1))
        return scores.transpose(1, 2)


class BBoxHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query_projector = nn.Linear(d_model, d_model * 2)
        self.xyz_projector = nn.Linear(3, d_model)
        self.feature_projector = nn.Identity()
        self.cross_attention = nn.MultiheadAttention(
            d_model * 2, num_heads=16, dropout=0.1, batch_first=True
        )
        self.bbox_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 6),
        )

    def _net(self, query_feats, ptc_feats, ptc_xyz, ptc_mask):
        query_feats = self.query_projector(query_feats)
        ptc_xyz = self.xyz_projector(ptc_xyz)
        ptc_feats = self.feature_projector(ptc_feats)
        ptc_feats_with_xyz = torch.cat([ptc_feats, ptc_xyz], dim=-1)

        attended = self.cross_attention(
            query_feats,
            ptc_feats_with_xyz,
            ptc_feats_with_xyz,
            key_padding_mask=ptc_mask,
            need_weights=False,
        )[0]

        pred = self.bbox_predictor(attended)
        center = pred[..., :3]
        dims = F.softplus(pred[..., 3:])
        return torch.cat([center, dims], dim=-1)

    def forward(self, query_feats, ptc_feats, ptc_xyz, ptc_mask=None):
        if self.training:
            return checkpoint(
                self._net, query_feats, ptc_feats, ptc_xyz, ptc_mask, use_reentrant=False
            )
        return self._net(query_feats, ptc_feats, ptc_xyz, ptc_mask)


class Locate3DDecoder(nn.Module):
    """Language-conditioned decoder producing (mask, text-alignment, bbox) per query."""

    def __init__(
        self,
        d_model=768,
        input_feat_dim=256,
        num_queries=256,
        num_decoder_layers=8,
        transformer_n_heads=12,
        transformer_dim_feedforward=3072,
        transformer_dropout=0.1,
        transformer_max_drop_path=0.0,
        transformer_use_checkpointing=True,
        freeze_text_encoder=True,
        text_encoder="clip",
    ):
        super().__init__()
        # Import here to allow models that are not using text encoders to avoid
        # the transformers dependency at import time.
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        assert text_encoder in ["clip", "clip-large"], "Only CLIP models are supported"
        self.clip_model = "openai/clip-vit-large-patch14"
        self.tokenizer = AutoTokenizer.from_pretrained(self.clip_model)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(self.clip_model)
        self.text_encoder_hidden_size = self.text_encoder.config.hidden_size
        self.max_tokens = 77
        self.freeze_text_encoder = freeze_text_encoder
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder_hidden_size, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1),
        )
        self.ptc_feat_projector = nn.Sequential(
            nn.Linear(input_feat_dim, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1),
        )

        self.pos_embed_3d = LearnedPosEmbeddings(dim=3, num_pos_feats=d_model)

        self.query_feat = nn.Embedding(num_queries, d_model)
        self.query_pos = nn.Embedding(num_queries, d_model)

        drop_paths = [
            x.item()
            for x in torch.linspace(0, transformer_max_drop_path, num_decoder_layers)
        ]
        self.decoder = nn.ModuleList(
            [
                TransformerModule(
                    d_model,
                    transformer_n_heads,
                    transformer_dim_feedforward,
                    transformer_dropout,
                    drop_paths[i],
                    transformer_use_checkpointing,
                )
                for i in range(num_decoder_layers)
            ]
        )

        self.mask_prediction_heads = nn.ModuleList(
            [MaskPredictionHead(d_model) for _ in range(num_decoder_layers)]
        )
        self.text_alignment_head = nn.ModuleList(
            [TextAlignmentHead(d_model, max_tokens=self.max_tokens) for _ in range(num_decoder_layers)]
        )
        self.bbox_head = BBoxHead(d_model)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.momentum = 0.1

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        return self

    def tokenize(self, captions, device):
        tokenized = self.tokenizer.batch_encode_plus(
            captions,
            padding="max_length",
            return_tensors="pt",
            max_length=self.max_tokens,
            truncation=True,
        ).to(device)
        return tokenized

    def forward(
        self,
        ptc_feats,          # (B, N, input_feat_dim) -- already padded
        ptc_xyz,            # (B, N, 3)
        ptc_key_padding_mask,  # (B, N) True for PAD
        captions,           # list[str] len B
    ):
        device = ptc_feats.device
        B = ptc_feats.shape[0]

        ptc_feats = self.ptc_feat_projector(ptc_feats)
        ptc_pos = self.pos_embed_3d(ptc_xyz)

        tokenized = self.tokenize(list(captions), device)
        text_attention_mask = tokenized.attention_mask  # (B, L)

        if self.freeze_text_encoder:
            with torch.no_grad():
                encoded_text = self.text_encoder(**tokenized)
        else:
            encoded_text = self.text_encoder(**tokenized)
        text_feats = self.text_projector(encoded_text.last_hidden_state)
        text_pos = torch.zeros_like(text_feats)

        # query_mask for self-attention: concat[query_mask(False), text_pad_mask]
        text_pad_mask = text_attention_mask.ne(1).bool()  # True where padded
        query_false = torch.zeros(B, self.num_queries, dtype=torch.bool, device=device)
        query_mask = torch.cat([query_false, text_pad_mask], dim=1)

        query_feats = self.query_feat.weight.unsqueeze(0).repeat(B, 1, 1)
        query_pos = self.query_pos.weight.unsqueeze(0).repeat(B, 1, 1)

        predictions_class = []
        predictions_mask = []
        predictions_boxes = []

        for i in range(self.num_decoder_layers):
            query_feats, text_feats, ptc_feats = self.decoder[i](
                query_feats,
                query_pos,
                text_feats,
                text_pos,
                ptc_feats,
                ptc_pos,
                query_mask,
                ptc_key_padding_mask,
            )

            mask = self.mask_prediction_heads[i](query_feats, ptc_feats)
            text_alignment = self.text_alignment_head[i](query_feats)
            bbox = self.bbox_head(query_feats, ptc_feats, ptc_xyz, ptc_mask=ptc_key_padding_mask)
            bbox = box_cxcyczwhd_to_xyzxyz_jit(bbox)

            predictions_class.append(text_alignment)
            predictions_mask.append(mask)
            predictions_boxes.append(bbox)

        out = {
            "text_attn_mask": tokenized.attention_mask.ne(1).bool(),
            "tokenized": tokenized,
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "pred_boxes": predictions_boxes[-1],
            "aux_outputs": [
                {
                    "pred_logits": c,
                    "pred_masks": m,
                    "pred_boxes": b,
                }
                for c, m, b in zip(
                    predictions_class[:-1],
                    predictions_mask[:-1],
                    predictions_boxes[:-1],
                )
            ],
        }
        return out
