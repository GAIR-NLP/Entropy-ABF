import torch
import math
from transformers.models.llama import modeling_llama, configuration_llama


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_old(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_scale(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    scale = ((position_ids + 1)[:, None, :, None].log() / math.log(4096)).to(q.dtype)
    scale[scale < 1] = 1
    return q_embed * scale, k_embed


def _init_rope_new(self):
    if self.config.rope_scaling is None:
        self.rotary_emb = modeling_llama.LlamaRotaryEmbedding(
            self.head_dim, max_position_embeddings=self.max_position_embeddings
        )
    else:
        scaling_type = self.config.rope_scaling["type"]
        scaling_factor = self.config.rope_scaling["factor"]
        if scaling_type == "linear":
            self.rotary_emb = modeling_llama.LlamaLinearScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
            )
        elif scaling_type == "dynamic":
            self.rotary_emb = modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
            )
        elif scaling_type == "eabf":
            self.rotary_emb = modeling_llama.LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=500000,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")


def _rope_scaling_validation_new(self):
    """
    Validate the `rope_scaling` configuration.
    """
    if self.rope_scaling is None:
        return

    if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
        raise ValueError(
            "`rope_scaling` must be a dictionary with with two fields, `name` and `factor`, "
            f"got {self.rope_scaling}"
        )
    rope_scaling_type = self.rope_scaling.get("type", None)
    rope_scaling_factor = self.rope_scaling.get("factor", None)
    if rope_scaling_type is None or rope_scaling_type not in [
        "linear",
        "dynamic",
        "eabf",  # support Entropy-Aware abf
    ]:
        raise ValueError(
            f"`rope_scaling`'s name field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
        )
    if (
        rope_scaling_factor is None
        or not isinstance(rope_scaling_factor, float)
        or rope_scaling_factor <= 1.0
    ):
        raise ValueError(
            f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}"
        )


def apply_eabf(model, count=2):
    """Replace scaled attention with original attention"""
    layers = model.model.layers

    for i in range(count):
        layers[i].self_attn.apply_rotary_pos_emb = apply_rotary_pos_emb_old


modeling_llama.apply_rotary_pos_emb = apply_rotary_pos_emb_scale
modeling_llama.LlamaAttention._init_rope = _init_rope_new
configuration_llama.LlamaConfig._rope_scaling_validation = _rope_scaling_validation_new
