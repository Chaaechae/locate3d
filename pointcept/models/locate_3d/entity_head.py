"""
EntityHead: per-CLIP-token classifier that turns a raw caption into
the per-entity ``positive_map`` the 0h Locate3DSegDetector consumes.

Today the SegDetector reads ``positive_map`` directly from the
annotation: each entity's caption-token positions are pre-labelled by
the ScanRefer / ScanEnts3D pipeline. For deployment on raw user
queries (no annotation) we need the model to *predict* that map. This
module is the predictor.

Architecture (small):
    CLIP-L text encoder (frozen, lives in the SegDetector)
    -> hidden states (B, T, 768)
    -> EntityHead:
         N TransformerEncoderLayer  (re-attend within caption)
         Linear(768, max_entities + 1)
    -> per-token logits (B, T, K+1)

Loss (training): per-token CrossEntropy with ``ignore_index=-1`` for
non-entity / padding tokens.

Inference: argmax along the last dim, then scatter token -> entity to
build a ``(G, T)`` positive_map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityHead(nn.Module):
    """Predict which (sub-)entity each CLIP token belongs to.

    Args:
        d_model: hidden size (must match CLIP text encoder, 768 for L).
        max_entities: K. Output dim is K+1 (the +1 is the "no entity"
            class). Captions exceeding K entities are truncated at
            data-prep time (see prepare_entity_head_data.py).
        n_layers: number of TransformerEncoderLayers stacked on top of
            the CLIP hidden states. 1-2 is plenty -- this is a
            lightweight refinement step, not a full re-encoder.
        nhead, dim_feedforward, dropout: standard transformer knobs.
    """

    def __init__(
        self,
        d_model: int = 768,
        max_entities: int = 4,
        n_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_entities = max_entities
        self.n_classes = max_entities + 1  # +1 for "no entity"

        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, self.n_classes)

    def forward(self, text_feats, attention_mask=None):
        """
        Args:
            text_feats: (B, T, d_model) -- frozen CLIP text encoder
                last_hidden_state.
            attention_mask: (B, T) bool / int -- 1 for real tokens,
                0 for padding. If provided, padding tokens are masked
                out of self-attention (no information leaks from / to
                them).

        Returns:
            logits: (B, T, K+1)
        """
        h = text_feats
        # TransformerEncoderLayer expects ``src_key_padding_mask``: True
        # for positions that should be ignored (the inverse of
        # ``attention_mask``).
        if attention_mask is not None:
            key_padding_mask = ~(attention_mask.bool())
        else:
            key_padding_mask = None
        for layer in self.attn_layers:
            h = layer(h, src_key_padding_mask=key_padding_mask)
        logits = self.classifier(h)  # (B, T, K+1)
        return logits

    @staticmethod
    def loss(
        logits,
        token_labels,
        attention_mask=None,
        label_smoothing: float = 0.0,
        no_entity_weight: float = 0.25,
        supervise_no_entity: bool = True,
    ):
        """Per-token cross-entropy.

        With ``supervise_no_entity=True`` (the new default), -1 labels
        are mapped to the last class (K = max_entities = "no entity")
        so the model is actively trained to PREDICT "no entity" at
        articles / prepositions / filler tokens. Without this, the
        model is only told the entity tokens' classes -- it has no
        signal that "beside" / "the" / "is" should NOT be classified
        as an entity, so non-entity tokens get absorbed into the
        nearest entity (the user-reported failure mode).

        ``attention_mask``, if given, gates which positions contribute
        to the loss (padding tokens always ignored).

        ``no_entity_weight`` (default 0.25) downweights the K class so
        it doesn't drown out the rarer entity classes (~80% of valid
        tokens are no-entity; 0.25 ≈ inverse frequency).

        ``supervise_no_entity=False`` reverts to the old
        ignore_index=-1 behavior for ablation / backward-compat.
        """
        B, T, K1 = logits.shape
        K = K1 - 1
        labels = token_labels.clone()
        if supervise_no_entity:
            # Map -1 (no-entity content) -> K (the explicit "no entity"
            # class), then ignore only padding.
            labels[labels == -1] = K
            if attention_mask is not None:
                labels[~attention_mask.bool()] = -100  # ignore padding
            weight = torch.ones(K1, device=logits.device, dtype=logits.dtype)
            weight[K] = no_entity_weight
            return F.cross_entropy(
                logits.reshape(B * T, K1),
                labels.reshape(B * T),
                ignore_index=-100,
                weight=weight,
                label_smoothing=label_smoothing,
            )
        else:
            return F.cross_entropy(
                logits.reshape(B * T, K1),
                labels.reshape(B * T),
                ignore_index=-1,
                label_smoothing=label_smoothing,
            )

    @torch.no_grad()
    def predict_positive_map(
        self,
        text_feats,
        attention_mask=None,
        return_n_entities: bool = False,
    ):
        """Inference helper: given CLIP text feats, return a (B, K, T)
        positive_map matrix matching the Locate3DSegDetector's input
        format. Tokens whose argmax is the "no entity" class (== K)
        contribute to no entity row. Trailing entity rows that received
        no token are zeroed but kept to preserve a stable shape.
        """
        logits = self.forward(text_feats, attention_mask=attention_mask)
        pred_token_class = logits.argmax(dim=-1)  # (B, T) in [0, K]
        B, T = pred_token_class.shape
        pos_map = torch.zeros(
            B, self.max_entities, T,
            device=logits.device, dtype=torch.float,
        )
        for k in range(self.max_entities):
            pos_map[:, k, :] = (pred_token_class == k).float()
        # Zero out padding columns even if argmax accidentally picked
        # an entity for them.
        if attention_mask is not None:
            pos_map = pos_map * attention_mask.unsqueeze(1).float()
        if return_n_entities:
            n_ent = (pos_map.sum(dim=-1) > 0).sum(dim=-1)  # (B,)
            return pos_map, n_ent
        return pos_map
