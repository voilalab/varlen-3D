# Copyright (c) 2025, Biao Zhang.
#
# HyperAutoEncoder — variable-length encoding on top of a frozen base model.
#
# Architecture overview:
#   Encoder: learnable queries (0..latent_n-1) cross-attend the ordered shape
#            latents with RoPE, then refine with RoPE self-attention.
#   Decoder: fixed sinusoidal positional queries cross-attend the prefix of
#            hyper-tokens (keys augmented with sinusoidal PE), then refine with
#            RoPE self-attention.  No residual on the cross-attention so the
#            decoder cannot bypass the context.

import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from flash_attn import flash_attn_kvpacked_func

from .utils import PreNorm, FeedForward


# ── RoPE utilities ────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Stores inverse frequencies; call get_cos_sin(seq_len, device) to obtain the
    per-position cosine/sine matrices used by _apply_rope.
    """
    def __init__(self, dim_head: int):
        super().__init__()
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim_head, 2).float() / dim_head)
        )
        self.register_buffer('inv_freq', inv_freq)

    def get_cos_sin(self, seq_len: int, device: torch.device):
        t     = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # N × (D/2)
        emb   = torch.cat([freqs, freqs], dim=-1)           # N × D
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
    """
    x   : B × N × H × D_head
    cos : N × D_head  (broadcast over B and H)
    sin : N × D_head
    """
    cos = cos.unsqueeze(0).unsqueeze(2)   # 1 × N × 1 × D
    sin = sin.unsqueeze(0).unsqueeze(2)
    return x * cos + _rotate_half(x) * sin


# ── Sinusoidal positional encoding (fixed) ───────────────────────────────────

def _sinusoidal_pe(n: int, dim: int) -> torch.Tensor:
    pe  = torch.zeros(n, dim)
    pos = torch.arange(0, n).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ── Flash Attention with optional per-call RoPE ───────────────────────────────

class RopeAttention(nn.Module):
    """
    Standard multi-head attention (Flash-Attention backend) with optional RoPE.

    rope_q / rope_k : (cos, sin) tuples of shape (N × D_head).
                      Pass None to skip rotation on that side.
    causal          : if True, applies a causal (lower-triangular) mask so
                      position k can only attend to positions 0..k-1.
                      Used in encoder self-attention to force progressive
                      information packing across hyper-tokens.
    """
    def __init__(self,
                 query_dim:   int,
                 context_dim: int  = None,
                 heads:       int  = 8,
                 dim_head:    int  = 64,
                 causal:      bool = False):
        super().__init__()
        inner_dim    = dim_head * heads
        context_dim  = context_dim or query_dim
        self.heads   = heads
        self.causal  = causal
        self.to_q    = nn.Linear(query_dim,   inner_dim,     bias=False)
        self.to_kv   = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out  = nn.Linear(inner_dim,   query_dim)

    def forward(self, x, context=None, rope_q=None, rope_k=None):
        h          = self.heads
        kv_input   = context if context is not None else x

        q  = rearrange(self.to_q(x),        'b n (h d) -> b n h d', h=h)
        kv = rearrange(self.to_kv(kv_input),'b n (p h d) -> b n p h d', h=h, p=2)
        k, v = kv[:, :, 0], kv[:, :, 1]    # each B × N × H × D

        if rope_q is not None:
            q = _apply_rope(q, *rope_q)
        if rope_k is not None:
            k = _apply_rope(k, *rope_k)

        kv  = torch.stack([k, v], dim=2)    # B × N_kv × 2 × H × D
        out = flash_attn_kvpacked_func(q.bfloat16(), kv.bfloat16(), causal=self.causal)
        out = out.to(x.dtype)
        return self.to_out(rearrange(out, 'b n h d -> b n (h d)'))


# ── Hyper-Encoder ─────────────────────────────────────────────────────────────

class HyperEncoder(nn.Module):
    """
    Maps ordered shape latents (B × latent_n × latent_dim) →
    ordered hyper-latents (B × latent_n × latent_dim).

    Cross-attention: learnable query i attends to shape latent i via RoPE, so
    hyper-token i primarily captures information at position i of the shape
    latent — the key mechanism that creates an ordered, prefix-truncatable
    representation.

    Self-attention: causal + RoPE.  Token k may only attend to tokens 0..k-1,
    forcing a progressive information hierarchy: token 0 holds the coarsest
    global summary (visible to all later tokens), and each subsequent token
    adds finer detail conditioned on what came before.  Without causal masking,
    bidirectional self-attention collapses every token into the same global
    average, making any prefix equally informative → flat RD curve.
    """
    def __init__(self,
                 latent_n:   int = 1024,
                 latent_dim: int = 32,
                 working_dim:int = 512,
                 depth:      int = 8,
                 dim_head:   int = 64):
        super().__init__()
        self.latent_n = latent_n
        heads = working_dim // dim_head

        self.input_proj = nn.Linear(latent_dim, working_dim)
        self.queries    = nn.Embedding(latent_n, working_dim)
        self.rope       = RotaryEmbedding(dim_head)

        # Cross-attention with RoPE (PreNorm on Q; separate norm on context)
        self.cross_attn   = PreNorm(working_dim,
                                    RopeAttention(working_dim, working_dim, heads, dim_head))
        self.cross_norm_c = nn.LayerNorm(working_dim)
        self.cross_ff     = PreNorm(working_dim, FeedForward(working_dim))

        # Self-attention layers with RoPE + causal mask.
        # Causal masking forces token k to summarise only positions 0..k-1,
        # creating a progressive information hierarchy:
        #   token 0 → coarsest global summary (seen by all later tokens)
        #   token k → fine detail for position k (conditioned on 0..k-1)
        # Without this, bidirectional self-attention collapses all tokens into
        # the same global average → any prefix carries identical information →
        # flat RD curve.
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(working_dim,
                        RopeAttention(working_dim, heads=heads, dim_head=dim_head,
                                      causal=True)),
                PreNorm(working_dim, FeedForward(working_dim)),
            ])
            for _ in range(depth)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(working_dim),
            nn.Linear(working_dim, latent_dim),
        )

    def forward(self, shape_latents: torch.Tensor) -> torch.Tensor:
        # shape_latents : B × latent_n × latent_dim
        B, N, _ = shape_latents.shape

        context = self.input_proj(shape_latents)               # B × N × working_dim
        x       = repeat(self.queries.weight, 'n d -> b n d', b=B)

        # Both sequences share the same positional semantics (0 .. latent_n-1)
        cos, sin = self.rope.get_cos_sin(N, shape_latents.device)

        # Cross-attention with RoPE; standard residual
        x = self.cross_attn(x,
                            context=self.cross_norm_c(context),
                            rope_q=(cos, sin),
                            rope_k=(cos, sin)) + x
        x = self.cross_ff(x) + x

        # Self-attention with RoPE
        for attn, ff in self.layers:
            x = attn(x, rope_q=(cos, sin), rope_k=(cos, sin)) + x
            x = ff(x) + x

        return self.output_proj(x)          # B × latent_n × latent_dim


# ── Hyper-Decoder ─────────────────────────────────────────────────────────────

class HyperDecoder(nn.Module):
    """
    Maps a prefix of hyper-latents (B × L × latent_dim, L ≤ latent_n) →
    full shape latent reconstruction (B × latent_n × latent_dim).

    Queries: fixed sinusoidal PE at positions 0..latent_n-1.  No learnable
    query biases means there is no shortcut that lets the decoder ignore the
    context — it must read from the prefix to produce any output.

    Keys:    projected hyper-tokens + sinusoidal PE, so the decoder knows which
    position each hyper-token came from when routing attention.

    RoPE is applied only in the subsequent self-attention layers (not in cross-
    attention, where Q and K come from different sequences of different lengths).
    """
    def __init__(self,
                 latent_n:   int = 1024,
                 latent_dim: int = 32,
                 working_dim:int = 512,
                 depth:      int = 8,
                 dim_head:   int = 64):
        super().__init__()
        self.latent_n = latent_n
        heads = working_dim // dim_head

        self.input_proj = nn.Linear(latent_dim, working_dim)

        # Fixed sinusoidal PE: one vector per latent_n position, dim = working_dim.
        # Shared between decoder queries (all latent_n positions) and hyper-token
        # keys (first L positions), ensuring both sides speak the same positional
        # language in the cross-attention.
        self.register_buffer('sin_pe', _sinusoidal_pe(latent_n, working_dim))

        self.rope = RotaryEmbedding(dim_head)

        # Cross-attention: no residual (see class docstring)
        # Separate LayerNorms for Q (positional) and C (hyper-tokens).
        self.cross_norm_q = nn.LayerNorm(working_dim)
        self.cross_norm_c = nn.LayerNorm(working_dim)
        self.cross_attn   = RopeAttention(working_dim, working_dim, heads, dim_head)
        self.cross_ff     = PreNorm(working_dim, FeedForward(working_dim))

        # Self-attention layers with RoPE
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(working_dim, RopeAttention(working_dim, heads=heads, dim_head=dim_head)),
                PreNorm(working_dim, FeedForward(working_dim)),
            ])
            for _ in range(depth)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(working_dim),
            nn.Linear(working_dim, latent_dim),
        )

    def forward(self, hyper_tokens: torch.Tensor) -> torch.Tensor:
        # hyper_tokens : B × L × latent_dim  (L ≤ latent_n)
        B, L, _ = hyper_tokens.shape

        # Context: projected hyper-tokens + sinusoidal PE for positions 0..L-1
        # so the cross-attention can distinguish which position each key came from.
        context = self.input_proj(hyper_tokens) + self.sin_pe[:L]   # B × L × working_dim

        # Queries: sinusoidal PE at all latent_n output positions (no learnable bias)
        x = self.sin_pe.unsqueeze(0).expand(B, -1, -1)              # B × latent_n × working_dim

        # Cross-attention: no residual — x must derive all shape content from context.
        # (Adding x back would reintroduce the PE as a learnable-query shortcut.)
        x = self.cross_attn(self.cross_norm_q(x),
                            context=self.cross_norm_c(context))      # no + x
        x = self.cross_ff(x) + x

        # Self-attention with RoPE — refines reconstruction while preserving order
        cos, sin = self.rope.get_cos_sin(self.latent_n, hyper_tokens.device)
        for attn, ff in self.layers:
            x = attn(x, rope_q=(cos, sin), rope_k=(cos, sin)) + x
            x = ff(x) + x

        return self.output_proj(x)          # B × latent_n × latent_dim


# ── HyperAutoEncoder ──────────────────────────────────────────────────────────

class HyperAutoEncoder(nn.Module):
    def __init__(self,
                 latent_n:          int  = 1024,
                 latent_dim:        int  = 32,
                 working_dim:       int  = 512,
                 encoder_depth:     int  = 8,
                 decoder_depth:     int  = 8,
                 dim_head:          int  = 64,
                 min_prefix:        int  = 64,
                 bias_small_prefix: bool = False):
        super().__init__()
        self.latent_n          = latent_n
        self.min_prefix        = min_prefix
        self.bias_small_prefix = bias_small_prefix

        self.encoder = HyperEncoder(
            latent_n=latent_n, latent_dim=latent_dim,
            working_dim=working_dim, depth=encoder_depth, dim_head=dim_head,
        )
        self.decoder = HyperDecoder(
            latent_n=latent_n, latent_dim=latent_dim,
            working_dim=working_dim, depth=decoder_depth, dim_head=dim_head,
        )

    def sample_prefix_length(self) -> int:
        """Sample prefix length L with p(L) ∝ L² on [min_prefix, latent_n].

        Compared to uniform sampling, L² heavily favours longer prefixes,
        substantially reducing the gradient deficit for later tokens:

            token k visibility = P(L > k) = 1 - ((k³ - min³) / (max³ - min³))

        e.g. with min=1, max=1024:
            k=512  → 87.5%  (vs 50% uniform)
            k=900  → 24.2%  (vs 12% uniform)
            k=1023 → 0.3%   (vs 0.1% uniform)

        Inverse CDF: L = (min³ + u·(max³ - min³))^(1/3), u ~ Uniform[0,1].
        """
        lo3 = self.min_prefix ** 3
        hi3 = self.latent_n  ** 3
        u   = torch.rand(1).item()
        L   = round((lo3 + u * (hi3 - lo3)) ** (1.0 / 3.0))
        return max(self.min_prefix, min(int(L), self.latent_n))

    def encode(self, shape_latents: torch.Tensor) -> torch.Tensor:
        """Returns full hyper-latent: B × latent_n × latent_dim."""
        return self.encoder(shape_latents)

    def decode(self, hyper_tokens: torch.Tensor) -> torch.Tensor:
        """Reconstructs shape latent from a prefix of hyper-tokens."""
        return self.decoder(hyper_tokens)

    def forward(self, shape_latents: torch.Tensor, prefix_length: int = -1):
        """
        Args:
            shape_latents : B × latent_n × latent_dim
            prefix_length : tokens to use for decoding; sampled if < 0
        Returns:
            reconstructed : B × latent_n × latent_dim
            hyper_latents : B × latent_n × latent_dim
            prefix_length : int
        """
        hyper_latents = self.encode(shape_latents)

        if prefix_length < 0:
            prefix_length = self.sample_prefix_length()

        reconstructed = self.decode(hyper_latents[:, :prefix_length])
        return reconstructed, hyper_latents, prefix_length
