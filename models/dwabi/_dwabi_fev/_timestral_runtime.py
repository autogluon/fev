"""Memory-bounded inference for the pinned Timestral split-attention SDK."""

import torch
from timestral.layers import _apply_rotary, _sdpa
from timestral.masks import (
    compute_enc_key_validity,
    enc_causal_mask,
    group_first_observed_patch,
)
from timestral.model import ROLE_KNOWN_COV, ROLE_TARGET
from timestral.scaler import denormalize_quantiles
from torch import nn
from torch.nn import functional as F

ATTENTION_SCORE_LIMIT = 16 * 2**20


def _attend(query, key, value, mask):
    batches, queries, heads, _ = query.shape
    keys = key.shape[1]
    if batches * heads * queries * keys <= ATTENTION_SCORE_LIMIT:
        return _sdpa(query, key, value, mask)
    capacity = ATTENTION_SCORE_LIMIT // (heads * keys)
    if capacity < 1:
        raise ValueError("Timestral attention cannot fit one query with its full key set")
    batch_step = max(1, capacity // queries)
    parts = []
    for row in range(0, batches, batch_step):
        stop = min(row + batch_step, batches)
        rows = slice(row, stop)
        query_step = capacity // (stop - row)
        pieces = []
        for start in range(0, queries, query_step):
            indices = slice(start, min(start + query_step, queries))
            local_mask = None if mask is None else mask[rows, :, indices, :]
            pieces.append(_sdpa(query[rows, indices], key[rows], value[rows], local_mask))
        parts.append(torch.cat(pieces, dim=1))
    return torch.cat(parts, dim=0)


def _time_masks(valid, context, extension, encoder_known):
    n = valid.shape[-1]
    encoder = enc_causal_mask(valid)
    if encoder_known is not None:
        diagonal = torch.eye(n, dtype=torch.bool, device=valid.device)
        encoder = torch.where(encoder_known[:, None, None], valid[:, None, :] | diagonal, encoder)
    context_valid = valid[:, None, :context].expand(-1, extension, -1)
    decoder_context = F.pad(context_valid, (0, n - context), value=False)
    decoder_causal = valid.new_ones(extension, extension).tril()
    decoder = torch.cat((decoder_context, decoder_causal.expand(len(valid), -1, -1)), dim=-1)
    return encoder[:, None], decoder[:, None]


def _variate_mask(valid, extension, groups, target, known):
    # The SDK forecaster packs all variates of one item in contiguous rows.
    variates = valid.shape[0] // groups
    if variates == groups == 1:
        return None
    keys = torch.cat((valid, valid.new_ones(len(valid), extension)), dim=1).T.reshape(-1, groups, variates)
    allowed = keys[:, :, None, :].expand(-1, -1, variates, -1)
    if target is not None:
        target = target.reshape(groups, variates)
        known = known.reshape(groups, variates)
        firewall = (target[:, :, None] | ~target[:, None, :]) & (~known[:, :, None] | known[:, None, :])
        allowed = allowed & firewall
    diagonal = torch.eye(variates, dtype=torch.bool, device=valid.device)
    return (allowed | diagonal).reshape(-1, 1, variates, variates)


def _time_attention(layer, tokens, masks, cos, sin):
    query, key, value = layer._qkv(tokens)
    query, key = _apply_rotary(query, key, cos, sin)
    encoder, decoder = masks
    n = encoder.shape[-1]
    context = _attend(query[:, :n], key[:, :n], value[:, :n], encoder)
    horizon = _attend(query[:, n:], key, value, decoder)
    mixed = torch.cat((context, horizon), dim=1)
    if layer._gate is not None:
        mixed = mixed * layer._gate(tokens).sigmoid()
    return layer.wO(mixed)


def _variate_attention(layer, tokens, mask, groups):
    rows, steps, width = tokens.shape
    variates = rows // groups
    inputs = tokens.transpose(0, 1)
    query, key, value = (part.reshape(steps * groups, variates, layer.num_heads, layer.head_dim)
                         for part in layer._qkv(inputs))
    mixed = _attend(query, key, value, mask).reshape(steps, rows, width)
    if layer._gate is not None:
        mixed = mixed * layer._gate(inputs).sigmoid()
    return layer.wO(mixed.transpose(0, 1))


class BoundedTimestral(nn.Module):
    """Share the loaded checkpoint and keep every admissible attention edge."""

    def __init__(self, model):
        super().__init__()
        if not model.config.split_context_horizon_attn:
            raise ValueError("Bounded Timestral requires split context and horizon attention")
        self.model = model
        self.config = model.config

    def scale(self, inputs, observed):
        return self.model.scale(inputs, observed)

    def _decode(self, tokens, positions, time_masks, variate_mask, groups, n):
        transformer = self.model.transformer
        cos, sin = transformer.rope(tokens, positions)
        for block in transformer.layers:
            for normalizer, layer in zip(block.attn_norms, block.attns, strict=True):
                normalized = normalizer(tokens)
                if layer.is_time:
                    result = _time_attention(layer, normalized, time_masks, cos, sin)
                else:
                    result = _variate_attention(layer, normalized, variate_mask, groups)
                tokens = tokens + result
            tokens = tokens + block.mlp(block.mlp_norm(tokens))
        return self.model.head(transformer.out_norm(tokens)[:, n:])

    def predict_block(self, inputs, observed, group_ids, horizon_patches, role_ids=None,
                      known_future=None, known_future_mask=None):
        size = self.config.patch_size
        context = inputs.shape[-1] // size
        target = known = None
        if role_ids is not None and self.config.covariate_enabled and not bool((role_ids == ROLE_TARGET).all()):
            target = role_ids == ROLE_TARGET
            known = role_ids == ROLE_KNOWN_COV
            if known_future is None:
                extent = (len(inputs), horizon_patches * size)
                known_future = inputs.new_zeros(extent)
                known_future_mask = observed.new_zeros(extent)
            inputs = torch.cat((inputs, known_future), dim=-1)
            observed = torch.cat((observed, known_future_mask), dim=-1)

        scaled, location, scale = self.scale(inputs, observed)
        if known is not None:
            scaled, location, scale = self.model._known_cov_scaling(inputs, observed, scaled, location, scale, known)
        encoder = self.model.representation(scaled, observed)
        decoder = self.model.representation.forecast_tokens(len(encoder), horizon_patches)
        observed_patches = observed.unflatten(-1, (-1, size)).any(dim=-1)
        valid = compute_enc_key_validity(observed_patches, self.config.attend_missing_patches)
        n = valid.shape[-1]
        ids, group_index = torch.unique(group_ids, return_inverse=True)
        groups = ids.numel()
        first = group_first_observed_patch(observed_patches, group_index, groups)[group_index]
        positions = torch.cat((torch.arange(n, device=inputs.device),
                               torch.arange(context, context + horizon_patches, device=inputs.device)))
        predictions = self._decode(
            torch.cat((encoder, decoder), dim=1), positions[None] - first[:, None],
            _time_masks(valid, context, horizon_patches, known),
            _variate_mask(valid, horizon_patches, groups, target, known), groups, n,
        )
        anchor = (context - 1) * size
        return denormalize_quantiles(predictions, location[:, anchor:anchor + 1, None],
                                     scale[:, anchor:anchor + 1, None], self.config.use_asinh)
