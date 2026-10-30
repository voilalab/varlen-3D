import math
from functools import wraps

from jaxtyping import Float
import numpy as np

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from einops import rearrange, repeat

from torch_cluster import fps

from timm.layers import DropPath

from external.flextok.token_dropout import MaskedNestedDropout


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def create_causal_mask(seq_len, device):
    """Create a lower triangular causal mask where position i can attend to j if i >= j."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask


def precompute_rope_frequencies(dim, max_seq_len, theta=10000.0):
    """Precompute cos/sin frequencies for Rotary Position Embeddings.

    Args:
        dim: Dimension per head (must be even).
        max_seq_len: Maximum sequence length to precompute for.
        theta: Base for frequency computation.

    Returns:
        Tuple of (cos, sin) tensors, each of shape (max_seq_len, dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)  # (max_seq_len, dim // 2)
    return freqs.cos(), freqs.sin()


def apply_rotary_pos_emb(x, cos, sin):
    """Apply rotary position embedding to a tensor.

    Works with any number of leading batch dimensions:
    e.g. (b*h, n, d) or (b, h, n, d).

    Args:
        x: Input tensor with last two dims (..., seq_len, dim_head).
        cos: Precomputed cosines of shape (max_seq_len, dim_head // 2).
        sin: Precomputed sines of shape (max_seq_len, dim_head // 2).

    Returns:
        Rotated tensor of same shape as x.
    """
    seq_len = x.shape[-2]
    d_half = x.shape[-1] // 2
    if cos.dim() == 3:
        # per-sample tables (B, n, d/2); x must be (B, h, n, d)
        assert x.dim() == 4, "per-sample RoPE tables need (batch, heads, seq, dim) inputs"
        cos = cos[:, :seq_len].unsqueeze(1)
        sin = sin[:, :seq_len].unsqueeze(1)
    else:
        cos = cos[:seq_len]
        sin = sin[:seq_len]
    # Broadcast cos/sin to match leading dims of x
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    x1, x2 = x[..., :d_half], x[..., d_half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def rope_tables_at(positions, dim_head, theta=10000.0):
    """cos/sin RoPE tables (len(positions), dim_head // 2) at arbitrary (float) positions."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim_head, 2, device=positions.device).float() / dim_head))
    angles = positions.float().unsqueeze(-1) * freqs  # (..., dim_head // 2): positions may be (n,) or (B, n)
    return angles.cos(), angles.sin()


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, drop_path_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))


class LegacyAttention(nn.Module):
    """Original manual attention implementation. Kept for backward compatibility and testing."""
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, drop_path_rate=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context=None, mask=None, rope_cos=None, rope_sin=None, rope_q=None, rope_k=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # Apply RoPE if provided (shared table for self-attention, or separate
        # (cos, sin) tables for queries and keys to bind cross-attention positions)
        if rope_cos is not None:
            q = apply_rotary_pos_emb(q, rope_cos, rope_sin)
            k = apply_rotary_pos_emb(k, rope_cos, rope_sin)
        if rope_q is not None:
            q = apply_rotary_pos_emb(q, *rope_q)
        if rope_k is not None:
            k = apply_rotary_pos_emb(k, *rope_k)

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            # Handle different mask shapes
            if mask.dim() == 2:  # (seq_len, seq_len)
                mask = repeat(mask, 'i j -> (b h) i j', b=x.shape[0], h=h)
            elif mask.dim() == 3:  # (batch, seq_len, seq_len)
                mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            else:  # Original code path for 1D masks
                mask = rearrange(mask, 'b ... -> b (...)')
                mask = repeat(mask, 'b j -> (b h) () j', h=h)

            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.drop_path(self.to_out(out))


class ModernAttention(nn.Module):
    """
    Modern attention implementation using PyTorch's scaled_dot_product_attention.

    Benefits over LegacyAttention:
    - Automatically uses Flash Attention 2 when available (PyTorch 2.0+)
    - 2-4x faster on modern GPUs
    - Significantly more memory efficient (up to 10x for long sequences)
    - Better numerical stability
    - Hardware-optimized kernels
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, drop_path_rate=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.dim_head = dim_head
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context=None, mask=None, rope_cos=None, rope_sin=None, rope_q=None, rope_k=None):
        B, N, _ = x.shape
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # Reshape to (batch, heads, seq_len, dim_head) for SDPA
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        # Apply RoPE if provided
        if rope_cos is not None:
            q = apply_rotary_pos_emb(q, rope_cos, rope_sin)
            k = apply_rotary_pos_emb(k, rope_cos, rope_sin)
        if rope_q is not None:
            q = apply_rotary_pos_emb(q, *rope_q)
        if rope_k is not None:
            k = apply_rotary_pos_emb(k, *rope_k)

        # Prepare attention mask for SDPA
        attn_mask = None
        if exists(mask):
            # Handle different mask shapes and convert to SDPA format
            if mask.dim() == 2:  # (seq_len, seq_len) - causal or custom attention mask
                # SDPA expects (batch, heads, seq_len_q, seq_len_kv) or broadcastable
                # For a 2D mask, it's (seq_len, seq_len), broadcast to all batches/heads
                attn_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            elif mask.dim() == 3:  # (batch, seq_len, seq_len)
                attn_mask = mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
            else:  # 1D or other - padding mask (batch, seq_len) or (batch, ..., seq_len)
                # Reshape to (batch, seq_len) if needed
                mask_reshaped = rearrange(mask, 'b ... -> b (...)')
                # For padding masks, SDPA expects shape (batch, seq_len_kv)
                # We need to broadcast this to attention mask format
                # Create a mask where True = attend, False = mask out
                attn_mask = mask_reshaped.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, seq_len_kv)

        # Use PyTorch's optimized scaled_dot_product_attention
        # This automatically uses Flash Attention 2 when available
        # Note: SDPA expects mask values where True = INCLUDE, not mask out
        # If your mask has False = attend, invert it
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,  # No dropout during attention
            is_causal=False  # We handle causality through attn_mask if needed
        )

        # Reshape back
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.drop_path(self.to_out(out))


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128, point_dim=3):
        super().__init__()

        assert hidden_dim % (2 * point_dim) == 0

        self.embedding_dim = hidden_dim
        self.point_dim = point_dim
        
        # Create basis vectors for each dimension
        freq_bands = self.embedding_dim // (2 * point_dim)
        e = torch.pow(2, torch.arange(freq_bands)).float() * np.pi
        
        # Create basis matrix: point_dim x hidden_dim
        basis_vectors = []
        for i in range(point_dim):
            basis_vector = torch.zeros(self.embedding_dim)
            start_idx = i * freq_bands
            basis_vector[start_idx:start_idx + freq_bands] = e
            basis_vectors.append(basis_vector)
        
        basis = torch.stack(basis_vectors)
        self.register_buffer('basis', basis)  # point_dim x hidden_dim

        # TODO: possible BUG - Dimension mismatch in PointEmbed MLP input size
        #  The embed() method (line 164) concatenates sin and cos, producing 2*embedding_dim features.
        #  Then forward() (line 169) concatenates this with the input, giving total features: 2*embedding_dim + point_dim
        #  However, this MLP is initialized to expect only embedding_dim + point_dim features.
        #  This causes "mat1 and mat2 shapes cannot be multiplied" errors.
        #  FIX: Change this line to: nn.Linear(2 * self.embedding_dim + point_dim, dim)
        #  However, this will break any pre-trained model weights, so coordinate with checkpoint compatibility.
        self.mlp = nn.Linear(self.embedding_dim + point_dim, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x point_dim
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2))  # B x N x C
        return embed


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                        + self.var - 1.0 - self.logvar,
                                        dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class BaseAutoEncoder(nn.Module):
    def __init__(
            self,
            *,
            decoder_depth=24,
            dim=512,
            queries_dim=512,
            num_inputs=2048,
            num_latents=512,
            heads=8,
            dim_head=64,
            weight_tie_layers=False,
            decoder_ff=False,
            point_dim=3,
            use_point_embed=True,
            use_nested_dropout=False,
            nested_dropout_size_sampling="uniform",
            use_causal_attention=False,
            use_legacy_attention=False,  # Set to True to use old implementation for testing
            ae_cross_heads=1,  # heads for the pooling/readout cross-attns (1 = original single-head)
            decoder_use_rope=False,
            encoder_depth=0,
            encoder_use_rope=False,
            encoder_use_pos_embed=False,
            varlen_mode="nested_dropout",  # "nested_dropout" | "prefix_truncation"
            use_rank_pe_latents=False,
            num_sink_tokens=0,
            query_binding="none",  # "none" | "rope": bind pooling query i to input i*stride and readout query j to latent j/stride
    ):
        super().__init__()

        assert query_binding in ("none", "rope", "rope_relative"), query_binding
        if query_binding == "rope_relative":
            assert not use_legacy_attention, "rope_relative needs ModernAttention (per-sample tables)"
        self.query_binding = query_binding
        self.bind_dim_head = dim // ae_cross_heads
        self.bind_stride = num_inputs / num_latents
        self.decoder_depth = decoder_depth
        self.encoder_depth = encoder_depth
        self.num_inputs = num_inputs
        self.num_latents = num_latents
        self.point_dim = point_dim
        self.use_point_embed = use_point_embed
        self.use_nested_dropout = use_nested_dropout
        self.use_causal_attention = use_causal_attention
        self.use_legacy_attention = use_legacy_attention
        self.decoder_use_rope = decoder_use_rope
        self.encoder_use_rope = encoder_use_rope
        self.encoder_use_pos_embed = encoder_use_pos_embed
        assert varlen_mode in ("nested_dropout", "prefix_truncation")
        self.varlen_mode = varlen_mode
        self.use_rank_pe_latents = use_rank_pe_latents
        self.num_sink_tokens = num_sink_tokens

        # Choose attention implementation
        Attention = LegacyAttention if use_legacy_attention else ModernAttention

        # Initialize nested dropout if enabled
        if use_nested_dropout:
            self.nested_dropout = MaskedNestedDropout(
                read_write_key="latents",
                dim=dim,
                size_sampling_mode=nested_dropout_size_sampling
            )
        else:
            self.nested_dropout = None

        assert dim % ae_cross_heads == 0, (dim, ae_cross_heads)
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads=ae_cross_heads, dim_head=dim // ae_cross_heads), context_dim=dim),
            PreNorm(dim, FeedForward(dim))
        ])

        if use_point_embed:
            # Use positional encoding point embedding (default)
            self.point_embed = PointEmbed(dim=dim, point_dim=point_dim)
        else:
            # Use identity or simple linear projection instead of point embedding
            # When point_dim == dim, just pass through (identity)
            # When point_dim != dim, use a linear layer to project to correct dimension
            # if point_dim != dim:
            #     self.point_embed = nn.Linear(point_dim, dim)
            # else:
            #     self.point_embed = nn.Identity()
            self.point_embed = nn.Identity()

        # Encoder self-attention layers (applied to input embeddings before cross-attention compression)
        self.encoder_layers = nn.ModuleList([])
        for i in range(encoder_depth):
            self.encoder_layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1)),
                PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
            ]))

        # Decoder self-attention layers (applied to latents before cross-attention decoding)
        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(decoder_depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        # Precompute RoPE frequencies for decoder self-attention layers
        # (sized to include sink tokens, which extend the latent sequence)
        if decoder_use_rope:
            rope_cos, rope_sin = precompute_rope_frequencies(dim_head, num_latents + num_sink_tokens)
            self.register_buffer('decoder_rope_cos', rope_cos)
            self.register_buffer('decoder_rope_sin', rope_sin)

        # Precompute RoPE frequencies for encoder self-attention layers
        if encoder_use_rope:
            rope_cos, rope_sin = precompute_rope_frequencies(dim_head, num_inputs)
            self.register_buffer('encoder_rope_cos', rope_cos)
            self.register_buffer('encoder_rope_sin', rope_sin)

        # Sinusoidal positional encoding added to input embeddings before encoder self-attention
        if encoder_use_pos_embed:
            self.register_buffer(
                'encoder_positional_encoding',
                create_sinusoidal_positional_encoding(num_inputs, dim)
            )

        # Rank PE: sinusoidal encoding of the latent INDEX (its coarse-to-fine
        # rank in the ordered sequence), added to the latents after dropout/
        # truncation so the decoder knows which rank each token carries.
        # Distinct from encoder_use_pos_embed, which encodes INPUT position.
        if use_rank_pe_latents:
            self.register_buffer(
                'latent_rank_pe',
                create_sinusoidal_positional_encoding(num_latents, dim)
            )

        # Attention sinks: learned latents PREPENDED after dropout/truncation.
        # Always visible to the decoder (and, under a causal mask, to every
        # real latent), never dropped, excluded from keep_k accounting. Give
        # the decoder a learned dataset-prior fallback at very low keep_k.
        if num_sink_tokens > 0:
            self.sink_tokens = nn.Parameter(torch.randn(num_sink_tokens, dim) * 0.02)

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads=ae_cross_heads, dim_head=dim // ae_cross_heads),
                                          context_dim=dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

    def _encoder_self_attend(self, x, mask=None):
        """May apply encoder self-attention (with optional RoPE) to input embeddings.

        This lets face embeddings exchange global information before being
        compressed into latents via cross-attention.

        Args:
            x: Input embeddings (B, N, dim).
            mask: Optional padding mask (B, 1, N) where True = valid.
        """
        if not self.encoder_layers:
            return x

        # Convert padding mask to self-attention mask if provided: mask KEYS only.
        # Masking padded queries too leaves those rows fully masked, which the
        # math SDPA backend turns into NaN that then leaks into the valid rows
        # through the next layer's values. Padded query rows are dropped as keys
        # downstream anyway, so they only need to stay finite.
        attn_mask = None
        if mask is not None:
            mask_1d = mask.squeeze(1)  # (B, N), True = valid face
            if getattr(self, "_legacy_encoder_mask", False):
                # pre-fix behaviour (queries masked too); kept only for the equivalence test
                attn_mask = mask_1d.unsqueeze(1) & mask_1d.unsqueeze(2)
            else:
                attn_mask = mask_1d.unsqueeze(1).expand(-1, mask_1d.shape[1], -1)  # (B, N_q, N_k)

        # Get RoPE frequencies if enabled
        rope_kwargs = {}
        if self.encoder_use_rope:
            rope_kwargs = {'rope_cos': self.encoder_rope_cos, 'rope_sin': self.encoder_rope_sin}

        for self_attn, self_ff in self.encoder_layers:
            x = self_attn(x, mask=attn_mask, **rope_kwargs) + x
            x = self_ff(x) + x
        return x

    def _may_posembed_selfattend(self, x, mask=None):
        """May add sinusoidal positional encoding, then apply encoder self-attention.

        Args:
            x: Input embeddings (B, N, dim).
            mask: Optional padding mask (B, 1, N) where True = valid.
        """
        if self.encoder_use_pos_embed:
            B, N, _ = x.shape
            pos_enc = self.encoder_positional_encoding[:N].unsqueeze(0).expand(B, -1, -1)
            x = x + pos_enc

        return self._encoder_self_attend(x, mask=mask)

    def _may_apply_nested_dropout(self, x, keep_k=None):
        """Apply nested dropout to latent tensors by converting to dict format."""
        if not self.use_nested_dropout or self.nested_dropout is None:
            return x

        # Convert tensor to dict format expected by MaskedNestedDropout
        # TODO: is this conversion to list inefficient?
        # x shape: [B, num_latents, dim]
        data_dict = {"latents": [x[i:i+1] for i in range(x.shape[0])]}  # List of [1, num_latents, dim] tensors

        # If keep_k is specified, add it to the dict for eval-time control
        if keep_k is not None:
            data_dict["eval_keep_k"] = [keep_k] * x.shape[0]  # Same keep_k for whole batch

        # Apply nested dropout
        data_dict = self.nested_dropout(data_dict)

        # Convert back to tensor format
        x = torch.cat(data_dict["latents"], dim=0)  # [B, num_latents, dim]
        return x

    def _pre_decode_transform(self, x):
        """Hook for subclasses to transform latents before decode processing.

        KL variants override this to apply projection from latent_dim to dim.
        Default implementation returns input unchanged.
        """
        return x

    def _may_truncate_prefix(self, x, keep_k=None):
        """Prefix truncation (varlen_mode="prefix_truncation"): actually SLICE
        the latent sequence to its first k tokens, HyperAutoEncoder-style,
        instead of replacing dropped tokens with a learned mask token.

        Training: k ~ Uniform[1, num_latents] (matches nested dropout's
        "uniform" sampling so dropout-vs-truncation comparisons are clean).
        Eval: slice to keep_k if given, else keep the full sequence.
        """
        if self.training:
            keep_k = int(torch.randint(1, x.shape[1] + 1, (1,)).item())
        if keep_k is not None:
            x = x[:, :int(keep_k)]
        return x

    def _may_dropout_causalmask_selfattend(self, x, keep_k=None):
        """May apply any of nested dropout / prefix truncation -> rank PE ->
        sink tokens -> causal mask -> self-attention to latents."""
        # Apply pre-decode transform (projection for KL variants, identity otherwise)
        x = self._pre_decode_transform(x)

        # Variable-length mechanism: mask-token dropout or true truncation
        if self.varlen_mode == "prefix_truncation":
            x = self._may_truncate_prefix(x, keep_k=keep_k)
        else:
            x = self._may_apply_nested_dropout(x, keep_k=keep_k)

        # Rank PE: tell the decoder which coarse-to-fine rank each latent has.
        # Applied after dropout/truncation; prefixes keep leading indices, so
        # index i is the rank both for masked and truncated sequences.
        if self.use_rank_pe_latents:
            x = x + self.latent_rank_pe[:x.shape[1]].to(x.dtype)

        # Prepend always-visible sink tokens (never dropped, not counted in k).
        # Prepended so that, under a causal mask, every real latent can see them.
        if self.num_sink_tokens > 0:
            sinks = self.sink_tokens.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([sinks, x], dim=1)

        # Create causal mask if needed
        causal_mask = None
        if self.use_causal_attention:
            seq_len = x.shape[1]  # num_latents (+ sinks)
            causal_mask = create_causal_mask(seq_len, x.device)

        # Get RoPE frequencies if enabled
        rope_kwargs = {}
        if self.decoder_use_rope:
            rope_kwargs = {'rope_cos': self.decoder_rope_cos, 'rope_sin': self.decoder_rope_sin}

        for self_attn, self_ff in self.layers:
            x = self_attn(x, mask=causal_mask, **rope_kwargs) + x
            x = self_ff(x) + x
        return x

    def _bind_kwargs(self, side, n_q, n_k, device, num_faces=None):
        """RoPE tables that bind cross-attention queries to context positions.

        side="pool": queries are the M latents at positions i*stride, keys the N
        faces at their index. side="read": queries are the output faces at their
        index, keys the latents at i*stride (sink tokens, if any, sit in front at
        negative positions). With the image-side binding, query i attends to
        input i by default, so full rate is the identity up to the packing.
        """
        if self.query_binding == "none":
            return {}
        # rope: one stride for every mesh (max faces / latents). rope_relative: each mesh's
        # own faces / latents, so every latent is anchored inside that mesh's face range.
        if self.query_binding == "rope_relative" and num_faces is not None:
            stride = (num_faces.to(device).float() / self.num_latents).clamp(min=1e-6).unsqueeze(1)  # (B, 1)
        else:
            stride = self.bind_stride
        if side == "pool":
            q_pos = torch.arange(n_q, device=device).float() * stride           # (n_q,) or (B, n_q)
            k_pos = torch.arange(n_k, device=device).float()
        else:
            q_pos = torch.arange(n_q, device=device).float()
            n_sinks = self.num_sink_tokens
            sink_pos = -torch.arange(n_sinks, 0, -1, device=device).float()
            lat_pos = torch.arange(n_k - n_sinks, device=device).float() * stride  # (n,) or (B, n)
            if lat_pos.dim() == 2:
                sink_pos = sink_pos.unsqueeze(0).expand(lat_pos.shape[0], -1)
            k_pos = torch.cat([sink_pos, lat_pos], dim=-1)
        return {"rope_q": rope_tables_at(q_pos, self.bind_dim_head),
                "rope_k": rope_tables_at(k_pos, self.bind_dim_head)}

    def _cross_attend_decode(self, x, queries_embeddings, num_faces=None):
        # cross attend from decoder queries to latents
        latents = self.decoder_cross_attn(queries_embeddings, context=x,
                                          **self._bind_kwargs("read", queries_embeddings.shape[1], x.shape[1], x.device, num_faces))

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        return latents


class KLMixin:
    """Mixin providing KL regularization components and methods.

    This mixin is used by KL autoencoder variants to add variational
    (KL-regularized) latent space. It provides:
    - Projection layers for mean/logvar parameterization
    - Reparameterization trick implementation
    - Override of _pre_decode_transform to project back from latent_dim to dim
    """

    def _init_kl_components(self, dim, latent_dim):
        """Initialize KL-specific layers. Call this in __init__ after super().__init__."""
        self.latent_dim = latent_dim
        self.proj = nn.Linear(latent_dim, dim)
        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

    def _apply_kl_transform(self, x):
        """Apply KL reparameterization trick.

        Args:
            x: Latent tensor of shape (batch, num_latents, dim)

        Returns:
            Tuple of (kl_loss, sample) where:
            - kl_loss: KL divergence loss of shape (batch,)
            - sample: Sampled latent of shape (batch, num_latents, latent_dim)
        """
        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)
        posterior = DiagonalGaussianDistribution(mean, logvar)
        return posterior.kl(), posterior.sample()

    def _pre_decode_transform(self, x):
        """Project from latent_dim back to dim before decode processing."""
        return self.proj(x)


class AutoEncoder(BaseAutoEncoder):
    def __init__(
            self,
            *,
            decoder_depth=24,
            dim=512,
            queries_dim=512,
            output_dim=1,
            num_inputs=2048,
            num_latents=512,
            heads=8,
            dim_head=64,
            weight_tie_layers=False,
            decoder_ff=False,
            point_dim=3,
            use_point_embed=True,
            use_nested_dropout=False,
            nested_dropout_size_sampling="uniform",
            use_causal_attention=False,
            use_legacy_attention=False,
            decoder_use_rope=False,
    ):
        super().__init__(
            decoder_depth=decoder_depth,
            dim=dim,
            queries_dim=queries_dim,
            num_inputs=num_inputs,
            num_latents=num_latents,
            heads=heads,
            dim_head=dim_head,
            weight_tie_layers=weight_tie_layers,
            decoder_ff=decoder_ff,
            point_dim=point_dim,
            use_point_embed=use_point_embed,
            use_nested_dropout=use_nested_dropout,
            nested_dropout_size_sampling=nested_dropout_size_sampling,
            use_causal_attention=use_causal_attention,
            use_legacy_attention=use_legacy_attention,
            decoder_use_rope=decoder_use_rope,
        )

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

    def encode(self, pc):
        # pc: B x N x point_dim
        B, N, D = pc.shape
        assert N == self.num_inputs

        ###### fps
        flattened = pc.view(B * N, D)

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened

        ratio = 1.0 * self.num_latents / self.num_inputs

        idx = fps(pos, batch, ratio=ratio)

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, self.point_dim)
        ######

        sampled_pc_embeddings = self.point_embed(sampled_pc)

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None) + sampled_pc_embeddings
        x = cross_ff(x) + x

        return x

    def decode(self, x, queries, keep_k=None):
        x = self._may_dropout_causalmask_selfattend(x, keep_k=keep_k)
        queries_embeddings = self.point_embed(queries)
        latents = self._cross_attend_decode(x, queries_embeddings)
        return self.to_outputs(latents)

    def forward(self, pc, queries, keep_k=None):
        x = self.encode(pc)
        o = self.decode(x, queries, keep_k=keep_k).squeeze(-1)
        return {'logits': o}


class LearnableQueryAutoEncoder(BaseAutoEncoder):
    def __init__(
            self,
            *,
            decoder_depth=24,
            dim=512,
            queries_dim=512,
            output_dim=1,
            num_inputs=2048,
            num_latents=512,
            heads=8,
            dim_head=64,
            weight_tie_layers=False,
            decoder_ff=False,
            point_dim=3,
            use_point_embed=True,
            use_nested_dropout=False,
            nested_dropout_size_sampling="uniform",
            use_causal_attention=False,
            use_legacy_attention=False,
            decoder_use_rope=False,
    ):
        super().__init__(
            decoder_depth=decoder_depth,
            dim=dim,
            queries_dim=queries_dim,
            num_inputs=num_inputs,
            num_latents=num_latents,
            heads=heads,
            dim_head=dim_head,
            weight_tie_layers=weight_tie_layers,
            decoder_ff=decoder_ff,
            point_dim=point_dim,
            use_point_embed=use_point_embed,
            use_nested_dropout=use_nested_dropout,
            nested_dropout_size_sampling=nested_dropout_size_sampling,
            use_causal_attention=use_causal_attention,
            use_legacy_attention=use_legacy_attention,
            decoder_use_rope=decoder_use_rope,
        )

        # Learnable query set L
        self.learnable_queries = nn.Parameter(torch.randn(num_latents, dim))
        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

    def encode(self, pc):
        # pc: B x N x point_dim
        B, N, D = pc.shape
        assert N == self.num_inputs

        # Use learnable queries instead of FPS
        pc_embeddings = self.point_embed(pc)  # PosEmb(X)

        # Expand learnable queries to batch size
        L = self.learnable_queries.unsqueeze(0).expand(B, -1, -1)  # B x M x C

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(L, context=pc_embeddings, mask=None) + L
        x = cross_ff(x) + x

        return x

    def decode(self, x, queries, keep_k=None):
        x = self._may_dropout_causalmask_selfattend(x, keep_k=keep_k)
        queries_embeddings = self.point_embed(queries)
        latents = self._cross_attend_decode(x, queries_embeddings)
        return self.to_outputs(latents)

    def forward(self, pc, queries, keep_k=None):
        x = self.encode(pc)
        o = self.decode(x, queries, keep_k=keep_k).squeeze(-1)
        return {'logits': o}


def create_sinusoidal_positional_encoding(max_len: int, dim: int) -> Tensor:
    """Create sinusoidal positional encoding.

    Args:
        max_len: Maximum sequence length
        dim: Embedding dimension

    Returns:
        Tensor of shape (max_len, dim) with sinusoidal positional encodings
    """
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))

    pe = torch.zeros(max_len, dim)
    n_sin = (dim + 1) // 2  # ceil(dim/2) - number of sin columns
    n_cos = dim // 2        # floor(dim/2) - number of cos columns
    pe[:, 0::2] = torch.sin(position * div_term[:n_sin])
    pe[:, 1::2] = torch.cos(position * div_term[:n_cos])

    return pe


class LearnableQueryExplicitAutoEncoder(BaseAutoEncoder):
    def __init__(
            self,
            *,
            decoder_depth=24,
            dim=512,
            queries_dim=512,
            num_inputs=2048,
            num_latents=512,
            num_output_points=512,
            heads=8,
            dim_head=64,
            weight_tie_layers=False,
            decoder_ff=False,
            point_dim=3,
            use_point_embed=True,
            use_nested_dropout=False,
            nested_dropout_size_sampling="uniform",
            use_causal_attention=False,
            use_legacy_attention=False,
            decoder_use_rope=False,
            encoder_depth=0,
            encoder_use_rope=False,
            encoder_use_pos_embed=False,
            varlen_mode="nested_dropout",
            use_rank_pe_latents=False,
            num_sink_tokens=0,
    ):
        super().__init__(
            decoder_depth=decoder_depth,
            dim=dim,
            queries_dim=queries_dim,
            num_inputs=num_inputs,
            num_latents=num_latents,
            heads=heads,
            dim_head=dim_head,
            weight_tie_layers=weight_tie_layers,
            decoder_ff=decoder_ff,
            point_dim=point_dim,
            use_point_embed=use_point_embed,
            use_nested_dropout=use_nested_dropout,
            nested_dropout_size_sampling=nested_dropout_size_sampling,
            use_causal_attention=use_causal_attention,
            use_legacy_attention=use_legacy_attention,
            decoder_use_rope=decoder_use_rope,
            encoder_depth=encoder_depth,
            encoder_use_rope=encoder_use_rope,
            encoder_use_pos_embed=encoder_use_pos_embed,
            varlen_mode=varlen_mode,
            use_rank_pe_latents=use_rank_pe_latents,
            num_sink_tokens=num_sink_tokens,
        )

        self.num_output_points = num_output_points

        # Learnable query set L for encoder
        self.learnable_queries = nn.Parameter(torch.randn(num_latents, dim))

        # Separate learnable queries for decoder
        self.decoder_queries = nn.Parameter(torch.randn(num_output_points, point_dim))
        
        # Output 3D coordinates instead of scalar values
        self.to_outputs = nn.Linear(queries_dim, point_dim)

    def encode(
            self,
            pc: Float[Tensor, "batch_size max_num_inputs point_dim"],
            num_faces: Float[Tensor, "batch_size"] | None = None,
    ) -> Float[Tensor, "batch_size num_latents embedding_dim"]:
        # pc: B x N x point_dim
        B, N, D = pc.shape
        # assert N == self.num_inputs

        # Use learnable queries instead of FPS
        pc_embeddings = self.point_embed(pc)  # PosEmb(X)

        # Create attention mask if num_faces is provided
        mask = None
        if num_faces is not None:
            # Create boolean mask: True for valid positions, False for padded positions
            # Shape: (B, 1, N) where mask[i, 0, j] = True if j < num_faces[i]
            mask = (torch.arange(N, device=pc.device).unsqueeze(0) < num_faces.unsqueeze(1)).unsqueeze(1)

        # Optional positional encoding + encoder self-attention
        pc_embeddings = self._may_posembed_selfattend(pc_embeddings, mask=mask)

        # Expand learnable queries to batch size
        L = self.learnable_queries.unsqueeze(0).expand(B, -1, -1)  # B x M x C

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(L, context=pc_embeddings, mask=mask) + L
        x = cross_ff(x) + x

        return x

    def decode(self, x, keep_k=None):
        B = x.shape[0]
        x = self._may_dropout_causalmask_selfattend(x, keep_k=keep_k)

        # Use separate learnable decoder queries
        decoder_queries = self.decoder_queries.unsqueeze(0).expand(B, -1, -1)  # B x num_output_points x point_dim

        queries_embeddings = self.point_embed(decoder_queries)
        # NOTE: only need queries in queries_embeddings which represent actual faces -- drop unnecessary ones downstream
        latents = self._cross_attend_decode(x, queries_embeddings)
        return self.to_outputs(latents)

    def forward(
            self,
            pc: Float[Tensor, "batch_size max_num_inputs point_dim"],
            num_faces: Float[Tensor, "batch_size"] | None = None,
            keep_k: int | None = None
    ):
        x = self.encode(pc, num_faces=num_faces)
        points = self.decode(x, keep_k=keep_k)  # B x num_output_points x point_dim
        return {'points': points}


class OrderPreservingExplicitAutoEncoder(BaseAutoEncoder):
    """Autoencoder that preserves input-output order correspondence.

    Uses fixed sinusoidal positional encoding as decoder queries (instead of learnable
    content queries), ensuring that output position i corresponds to input position i.

    The encoder is shared with LearnableQueryExplicitAutoEncoder — positional encoding
    on the encoder side is controlled by the encoder_use_pos_embed flag in BaseAutoEncoder.
    """

    def __init__(
            self,
            *,
            decoder_depth=24,
            dim=512,
            queries_dim=512,
            num_inputs=2048,
            num_latents=512,
            num_output_points=512,
            heads=8,
            dim_head=64,
            weight_tie_layers=False,
            decoder_ff=False,
            point_dim=3,
            use_point_embed=True,
            use_nested_dropout=False,
            nested_dropout_size_sampling="uniform",
            use_causal_attention=False,
            use_legacy_attention=False,
            decoder_use_rope=False,
            encoder_depth=0,
            encoder_use_rope=False,
            encoder_use_pos_embed=True,
            varlen_mode="nested_dropout",
            use_rank_pe_latents=False,
            num_sink_tokens=0,
            ae_cross_heads=1,
            query_binding="none",
            latent_quant_bits=0,      # 0 off; 4/8: per-token absmax uniform (straight-through in train); 16: bfloat16 cast
            latent_quant_train=False, # apply the quantizer in train mode too (else eval only)
    ):
        super().__init__(
            decoder_depth=decoder_depth,
            dim=dim,
            queries_dim=queries_dim,
            num_inputs=num_inputs,
            num_latents=num_latents,
            heads=heads,
            dim_head=dim_head,
            weight_tie_layers=weight_tie_layers,
            decoder_ff=decoder_ff,
            point_dim=point_dim,
            use_point_embed=use_point_embed,
            use_nested_dropout=use_nested_dropout,
            nested_dropout_size_sampling=nested_dropout_size_sampling,
            use_causal_attention=use_causal_attention,
            use_legacy_attention=use_legacy_attention,
            decoder_use_rope=decoder_use_rope,
            encoder_depth=encoder_depth,
            encoder_use_rope=encoder_use_rope,
            encoder_use_pos_embed=encoder_use_pos_embed,
            varlen_mode=varlen_mode,
            use_rank_pe_latents=use_rank_pe_latents,
            num_sink_tokens=num_sink_tokens,
            ae_cross_heads=ae_cross_heads,
            query_binding=query_binding,
        )

        self.num_output_points = num_output_points
        self.latent_quant_bits = int(latent_quant_bits)
        self.latent_quant_train = bool(latent_quant_train)

        # Learnable query set L for encoder (compresses N inputs to M latents)
        self.learnable_queries = nn.Parameter(torch.randn(num_latents, dim))

        # Fixed sinusoidal positional encoding for decoder queries
        self.register_buffer(
            'decoder_positional_encoding',
            create_sinusoidal_positional_encoding(num_output_points, dim)
        )

        # Output coordinates instead of scalar values
        self.to_outputs = nn.Linear(queries_dim, point_dim)

    def encode(
            self,
            pc: Float[Tensor, "batch_size max_num_inputs point_dim"],
            num_faces: Float[Tensor, "batch_size"] | None = None,
    ) -> Float[Tensor, "batch_size num_latents embedding_dim"]:
        # pc: B x N x point_dim
        B, N, D = pc.shape

        # Apply point embedding
        pc_embeddings = self.point_embed(pc)  # B x N x dim

        # Create attention mask if num_faces is provided
        mask = None
        if num_faces is not None:
            mask = (torch.arange(N, device=pc.device).unsqueeze(0) < num_faces.unsqueeze(1)).unsqueeze(1)

        # Optional positional encoding + encoder self-attention
        pc_embeddings = self._may_posembed_selfattend(pc_embeddings, mask=mask)

        # Expand learnable queries to batch size
        L = self.learnable_queries.unsqueeze(0).expand(B, -1, -1)  # B x M x dim

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(L, context=pc_embeddings, mask=mask,
                       **self._bind_kwargs("pool", L.shape[1], pc_embeddings.shape[1], L.device, num_faces)) + L
        x = cross_ff(x) + x

        return x

    def quantize_latents(self, x):
        """Scalar quantization of the transmitted latents (the pooled tokens, before the
        receiver-side latent self-attention). 16: bfloat16 cast. 4/8: symmetric uniform per
        token with an absmax scale (one extra float per token), straight-through gradient.
        Rate per kept token: bits x dim (+32 for the scale)."""
        bits = self.latent_quant_bits
        if bits <= 0 or (self.training and not self.latent_quant_train):
            return x
        if bits >= 16:
            return x.to(torch.bfloat16).to(x.dtype)
        levels = 2 ** (bits - 1) - 1
        scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / levels
        q = torch.round(x / scale).clamp(-levels, levels) * scale
        return x + (q - x).detach()

    def decode(self, x, keep_k=None, num_faces=None):
        B = x.shape[0]
        x = self.quantize_latents(x)
        x = self._may_dropout_causalmask_selfattend(x, keep_k=keep_k)

        # Use fixed positional encoding as decoder queries (NOT learnable content)
        # This ensures output position i asks for "content at position i"
        decoder_pos_queries = self.decoder_positional_encoding[:self.num_output_points]  # num_output_points x dim
        decoder_pos_queries = decoder_pos_queries.unsqueeze(0).expand(B, -1, -1)  # B x num_output_points x dim

        # Cross-attend from positional queries to compressed latents
        # Content flows through latents; position is determined by query position
        latents = self._cross_attend_decode(x, decoder_pos_queries, num_faces=num_faces)
        return self.to_outputs(latents)

    def forward(
            self,
            pc: Float[Tensor, "batch_size max_num_inputs point_dim"],
            num_faces: Float[Tensor, "batch_size"] | None = None,
            keep_k: int | None = None
    ):
        x = self.encode(pc, num_faces=num_faces)
        points = self.decode(x, keep_k=keep_k, num_faces=num_faces)  # B x num_output_points x point_dim
        return {'points': points}


def _flow_timestep_embedding(t, dim):
    """Sinusoidal embedding of t in [0, 1], scaled by 1000 (DiT convention).
    t: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000.0) * torch.arange(half, device=t.device).float() / half
    )
    args = t.float()[:, None] * 1000.0 * freqs[None]
    emb = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def _flow_modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class FlowBlock(nn.Module):
    """DiT-style block for the flow decoder: adaLN-zero self-attention and FF
    (modulated/gated by the timestep embedding, gates zero-init so the block
    starts as identity) plus plain-residual cross-attention to the latent
    conditioning (the conditioning path must stay open while gates are closed).
    """

    def __init__(self, dim, heads=8, dim_head=64, use_cross_attn=True):
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = ModernAttention(dim, heads=heads, dim_head=dim_head)

        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.norm_xq = nn.LayerNorm(dim)
            self.norm_xc = nn.LayerNorm(dim)
            self.cross_attn = ModernAttention(dim, dim, heads=heads, dim_head=dim_head)

        self.norm_ff = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = FeedForward(dim)

        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, context, t_emb, key_mask=None):
        sh_a, sc_a, g_a, sh_f, sc_f, g_f = self.ada(t_emb).chunk(6, dim=-1)
        x = x + g_a[:, None] * self.attn(
            _flow_modulate(self.norm_sa(x), sh_a, sc_a), mask=key_mask)
        if self.use_cross_attn:
            x = x + self.cross_attn(self.norm_xq(x), context=self.norm_xc(context))
        x = x + g_f[:, None] * self.ff(_flow_modulate(self.norm_ff(x), sh_f, sc_f))
        return x


class OrderPreservingFlowExplicitAutoEncoder(BaseAutoEncoder):
    """Flow-matching (generative) counterpart to OrderPreservingExplicitAutoEncoder.

    Encoder + latent processing are identical (learnable queries pool N face
    features to M latents; dropout/truncation + rank PE + sinks + causal
    self-attn all apply to the conditioning). The regression decoder is
    replaced by a conditional rectified-flow transformer over the face-feature
    sequence:

      Training:  flow_loss() -- sample t and noise, predict the velocity
                 v = pc - x0 conditioned on (x_t, t, latents); masked MSE over
                 valid faces. One decoder forward per step, no sampling.
      Inference: forward() integrates the ODE from noise with `flow_steps`
                 Euler steps and returns {'points': sample}, so the existing
                 validation path (frozen VQ -> conv decoder -> CE/chamfer)
                 consumes it unchanged.

    Order is preserved the same way as the regression decoder: face slot i of
    the noised sequence carries sinusoidal PE for position i, and the loss is
    per-position. The output is a SAMPLE, not the posterior mean -- per-face
    metrics are expected to punish this; that is the hypothesis under test.

    Intended for freeze_base training (the downstream CE loss cannot backprop
    through ODE sampling): only this module receives gradients, from flow_loss.
    """

    def __init__(
            self,
            *,
            decoder_depth=8,          # latent-conditioning self-attn blocks (as regression AE)
            flow_decoder_depth=8,     # DiT-style blocks in the flow decoder
            flow_steps=20,            # Euler steps at inference
            flow_t_sampling="logit_normal",  # "logit_normal" (SD3) | "mode" (SD3 mode sampling) | "uniform"
            flow_conditioning="cross_attn",  # "cross_attn" | "in_sequence" (FlexTok-register style)
            flow_mode_scale=0.25,     # s for flow_t_sampling="mode" (FlexTok: 0.25)
            flow_null_prob=0.0,       # CFG training: p of replacing condition w/ learned null
            flow_guidance=1.0,        # CFG scale at ODE sampling (1.0 = off)
            repa_weight=0.0,          # mesh REPA analog: align mid-decoder features to the CLEAN face features
            repa_layer=2,             # flow block index for the REPA feature tap
            dim=512,
            queries_dim=512,
            num_inputs=2048,
            num_latents=512,
            num_output_points=512,
            heads=8,
            dim_head=64,
            point_dim=3,
            use_point_embed=False,
            use_nested_dropout=False,
            nested_dropout_size_sampling="uniform",
            use_causal_attention=False,
            use_legacy_attention=False,
            decoder_use_rope=False,
            encoder_depth=0,
            encoder_use_rope=False,
            encoder_use_pos_embed=True,
            varlen_mode="nested_dropout",
            use_rank_pe_latents=False,
            num_sink_tokens=0,
            ae_cross_heads=1,
            query_binding="none",   # "none" | "rope" | "rope_relative": RoPE on the pooling queries (as the regression AE)
    ):
        super().__init__(
            decoder_depth=decoder_depth,
            dim=dim,
            queries_dim=queries_dim,
            num_inputs=num_inputs,
            num_latents=num_latents,
            heads=heads,
            dim_head=dim_head,
            point_dim=point_dim,
            use_point_embed=use_point_embed,
            use_nested_dropout=use_nested_dropout,
            nested_dropout_size_sampling=nested_dropout_size_sampling,
            use_causal_attention=use_causal_attention,
            use_legacy_attention=use_legacy_attention,
            decoder_use_rope=decoder_use_rope,
            encoder_depth=encoder_depth,
            encoder_use_rope=encoder_use_rope,
            encoder_use_pos_embed=encoder_use_pos_embed,
            varlen_mode=varlen_mode,
            use_rank_pe_latents=use_rank_pe_latents,
            num_sink_tokens=num_sink_tokens,
            ae_cross_heads=ae_cross_heads,
            query_binding=query_binding,
        )
        self.num_output_points = num_output_points
        self.flow_steps = flow_steps
        self.flow_t_sampling = flow_t_sampling
        assert flow_conditioning in ("cross_attn", "in_sequence")
        self.flow_conditioning = flow_conditioning
        self.flow_mode_scale = flow_mode_scale
        self.flow_null_prob = flow_null_prob
        self.flow_guidance = flow_guidance

        # Learned null condition (FlexTok: replace registers with a null token
        # w/ p during training; enables classifier-free guidance at sampling).
        self.flow_null_condition = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # in_sequence: conditioning latents are PREPENDED to the noised face
        # sequence (FlexTok-register style) and take part in joint self-attn;
        # role embeddings disambiguate conditioning tokens from face tokens.
        if flow_conditioning == "in_sequence":
            self.flow_role_emb = nn.Parameter(torch.zeros(2, dim))

        # Mesh REPA analog: cosine-align activations at flow block repa_layer
        # (computed from noised input + condition) to the CLEAN face features.
        # No external teacher (unlike image REPA/DINOv2): this keeps only the
        # anti-shortcut mechanism -- at high noise the clean features are
        # reachable only through the conditioning tokens.
        self.repa_weight = repa_weight
        self.repa_layer = repa_layer
        self.last_repa_loss = None
        if repa_weight > 0:
            self.flow_repa_proj = nn.Sequential(
                nn.Linear(dim, dim), nn.SiLU(),
                nn.Linear(dim, dim), nn.SiLU(),
                nn.Linear(dim, point_dim),
            )

        # The base class builds a regression cross-attention decoder we never
        # call. Remove it: its trainable-but-unused parameters would crash DDP
        # (find_unused_parameters=False) at the first backward.
        self.decoder_cross_attn = None
        self.decoder_ff = None

        # Encoder queries (same as OrderPreservingExplicitAutoEncoder)
        self.learnable_queries = nn.Parameter(torch.randn(num_latents, dim))

        # Flow decoder over face-feature tokens
        self.flow_input_proj = nn.Linear(point_dim, dim)
        self.register_buffer(
            'face_positional_encoding',
            create_sinusoidal_positional_encoding(num_output_points, dim)
        )
        self.t_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.flow_blocks = nn.ModuleList([
            FlowBlock(dim, heads=heads, dim_head=dim_head,
                      use_cross_attn=(flow_conditioning == "cross_attn"))
            for _ in range(flow_decoder_depth)
        ])
        self.flow_norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.flow_ada_out = nn.Linear(dim, 2 * dim)
        self.flow_proj_out = nn.Linear(dim, point_dim)
        nn.init.zeros_(self.flow_ada_out.weight)
        nn.init.zeros_(self.flow_ada_out.bias)
        nn.init.zeros_(self.flow_proj_out.weight)
        nn.init.zeros_(self.flow_proj_out.bias)

    # ── encoder (identical to OrderPreservingExplicitAutoEncoder) ─────────

    def encode(self, pc, num_faces=None):
        B, N, D = pc.shape
        pc_embeddings = self.point_embed(pc)

        mask = None
        if num_faces is not None:
            mask = (torch.arange(N, device=pc.device).unsqueeze(0) < num_faces.unsqueeze(1)).unsqueeze(1)

        pc_embeddings = self._may_posembed_selfattend(pc_embeddings, mask=mask)

        L = self.learnable_queries.unsqueeze(0).expand(B, -1, -1)
        cross_attn, cross_ff = self.cross_attend_blocks
        x = cross_attn(L, context=pc_embeddings, mask=mask,
                       **self._bind_kwargs("pool", L.shape[1], pc_embeddings.shape[1], L.device, num_faces)) + L
        x = cross_ff(x) + x
        return x

    def _condition(self, pc, num_faces=None, keep_k=None):
        """Encode + latent processing -> conditioning tokens (B, L(+S), dim)."""
        x = self.encode(pc, num_faces=num_faces)
        return self._may_dropout_causalmask_selfattend(x, keep_k=keep_k)

    # ── velocity prediction ────────────────────────────────────────────────

    def predict_velocity(self, x_t, t, condition, num_faces=None, feature_layer=None):
        """x_t: (B, N_out, point_dim), t: (B,), condition: (B, L, dim).
        feature_layer: if set, also return the face-token activations after that
        block index (REPA tap). Returns v or (v, features)."""
        B, N, _ = x_t.shape
        t_emb = self.t_mlp(_flow_timestep_embedding(t, self.face_positional_encoding.shape[1]).to(x_t.dtype))

        x = self.flow_input_proj(x_t) + self.face_positional_encoding[:N].to(x_t.dtype)

        # Key padding mask: padded face slots contain pure noise; keep valid
        # tokens from attending to them. Padded QUERIES still compute (their
        # loss is masked and downstream drops them). Shaped (B, N_q, N_kv)
        # because ModernAttention treats a 2-D mask as a (seq, seq) attention
        # mask, not a per-batch padding mask.
        key_mask = None
        if num_faces is not None:
            valid = (torch.arange(N, device=x_t.device).unsqueeze(0) < num_faces.unsqueeze(1))  # (B, N)

        features = None
        if self.flow_conditioning == "in_sequence":
            L = condition.shape[1]
            x = torch.cat([condition + self.flow_role_emb[0],
                           x + self.flow_role_emb[1]], dim=1)
            if num_faces is not None:
                # Conditioning tokens are always-valid keys; padded face slots
                # stay masked out for every query in the joint sequence.
                valid_full = torch.cat(
                    [torch.ones(B, L, dtype=torch.bool, device=x_t.device), valid], dim=1)
                key_mask = valid_full.unsqueeze(1).expand(-1, L + N, -1)
            for i, block in enumerate(self.flow_blocks):
                x = block(x, None, t_emb, key_mask=key_mask)
                if feature_layer is not None and i == feature_layer:
                    features = x[:, L:]
            x = x[:, L:]
        else:
            if num_faces is not None:
                key_mask = valid.unsqueeze(1).expand(-1, N, -1)  # every query attends valid keys only
            for i, block in enumerate(self.flow_blocks):
                x = block(x, condition, t_emb, key_mask=key_mask)
                if feature_layer is not None and i == feature_layer:
                    features = x

        shift, scale = self.flow_ada_out(t_emb).chunk(2, dim=-1)
        out = self.flow_proj_out(_flow_modulate(self.flow_norm_out(x), shift, scale))
        if feature_layer is not None:
            return out, features
        return out

    # ── training objective ─────────────────────────────────────────────────

    def flow_loss(self, pc, num_faces=None, keep_k=None):
        """Conditional rectified-flow loss, masked to valid faces."""
        B, N, _ = pc.shape
        condition = self._condition(pc, num_faces=num_faces, keep_k=keep_k)

        # CFG training: per-sample, replace the whole condition with the
        # learned null so the decoder also learns the unconditional field.
        if self.training and self.flow_null_prob > 0:
            drop = torch.rand(B, device=pc.device) < self.flow_null_prob
            null = self.flow_null_condition.to(condition.dtype).expand(
                B, condition.shape[1], -1)
            condition = torch.where(drop[:, None, None], null, condition)

        if self.flow_t_sampling == "logit_normal":
            t = torch.sigmoid(torch.randn(B, device=pc.device))
        elif self.flow_t_sampling == "mode":
            # SD3 mode sampling: concentrates t mid-trajectory w/ heavy tails.
            u = torch.rand(B, device=pc.device)
            s = self.flow_mode_scale
            t = 1 - u - s * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            t = torch.rand(B, device=pc.device)

        x0 = torch.randn_like(pc)
        t_ = t.to(pc.dtype)[:, None, None]
        x_t = (1 - t_) * x0 + t_ * pc
        v_target = pc - x0

        if self.repa_weight > 0:
            v_pred, feats = self.predict_velocity(
                x_t, t, condition, num_faces=num_faces, feature_layer=self.repa_layer)
        else:
            v_pred = self.predict_velocity(x_t, t, condition, num_faces=num_faces)

        mask2d = None
        if num_faces is not None:
            mask2d = (torch.arange(N, device=pc.device).unsqueeze(0)
                      < num_faces.unsqueeze(1)).float()

        sq_err = (v_pred.float() - v_target.float()) ** 2
        if mask2d is not None:
            mask = mask2d.unsqueeze(-1)
            loss = (sq_err * mask).sum() / (mask.sum() * pc.shape[-1]).clamp(min=1)
        else:
            loss = sq_err.mean()

        self.last_repa_loss = None
        if self.repa_weight > 0:
            proj = self.flow_repa_proj(feats)
            cos_dist = 1 - F.cosine_similarity(proj.float(), pc.float(), dim=-1)  # (B, N)
            if mask2d is not None:
                repa = (cos_dist * mask2d).sum() / mask2d.sum().clamp(min=1)
            else:
                repa = cos_dist.mean()
            self.last_repa_loss = repa.detach()
            loss = loss + self.repa_weight * repa
        return loss

    # ── inference (ODE sampling) ───────────────────────────────────────────

    @torch.no_grad()
    def sample(self, condition, num_faces=None, num_steps=None, guidance=None):
        """guidance: CFG scale (defaults to self.flow_guidance; 1.0 = off).
        Only meaningful when trained with flow_null_prob > 0."""
        num_steps = num_steps or self.flow_steps
        g = self.flow_guidance if guidance is None else guidance
        B, L = condition.shape[0], condition.shape[1]
        x = torch.randn(B, self.num_output_points, self.point_dim,
                        device=condition.device, dtype=condition.dtype)
        if g != 1.0:
            null = self.flow_null_condition.to(condition.dtype).expand(B, L, -1)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=x.device)
            v = self.predict_velocity(x, t, condition, num_faces=num_faces)
            if g != 1.0:
                v_null = self.predict_velocity(x, t, null, num_faces=num_faces)
                v = v_null + g * (v - v_null)
            x = x + v * dt
        return x

    def forward(self, pc, num_faces=None, keep_k=None):
        condition = self._condition(pc, num_faces=num_faces, keep_k=keep_k)
        points = self.sample(condition, num_faces=num_faces)
        return {'points': points}


class KLAutoEncoder(KLMixin, AutoEncoder):
    """KL-regularized autoencoder using FPS for latent selection.

    Inherits encoding logic from AutoEncoder and adds KL regularization
    via the KLMixin.
    """

    def __init__(self, *, latent_dim=64, **kwargs):
        super().__init__(**kwargs)
        self._init_kl_components(kwargs.get('dim', 512), latent_dim)

    def encode(self, pc):
        x = AutoEncoder.encode(self, pc)
        return self._apply_kl_transform(x)

    def forward(self, pc, queries, keep_k=None):
        kl, x = self.encode(pc)
        o = self.decode(x, queries, keep_k=keep_k).squeeze(-1)
        return {'logits': o, 'kl': kl}


class LearnableQueryKLAutoEncoder(KLMixin, LearnableQueryAutoEncoder):
    """KL-regularized autoencoder using learnable queries.

    Inherits encoding logic from LearnableQueryAutoEncoder and adds KL
    regularization via the KLMixin.
    """

    def __init__(self, *, latent_dim=64, **kwargs):
        super().__init__(**kwargs)
        self._init_kl_components(kwargs.get('dim', 512), latent_dim)

    def encode(self, pc):
        x = LearnableQueryAutoEncoder.encode(self, pc)
        return self._apply_kl_transform(x)

    def forward(self, pc, queries, keep_k=None):
        kl, x = self.encode(pc)
        o = self.decode(x, queries, keep_k=keep_k).squeeze(-1)
        return {'logits': o, 'kl': kl}


class LearnableQueryExplicitKLAutoEncoder(KLMixin, LearnableQueryExplicitAutoEncoder):
    """KL-regularized autoencoder with learnable queries and explicit output points.

    Inherits encoding logic from LearnableQueryExplicitAutoEncoder and adds KL
    regularization via the KLMixin.
    """

    def __init__(self, *, latent_dim=64, **kwargs):
        super().__init__(**kwargs)
        self._init_kl_components(kwargs.get('dim', 512), latent_dim)

    def encode(self, pc, num_faces=None):
        x = LearnableQueryExplicitAutoEncoder.encode(self, pc, num_faces=num_faces)
        return self._apply_kl_transform(x)

    def forward(self, pc, num_faces=None, keep_k=None):
        kl, x = self.encode(pc, num_faces=num_faces)
        points = self.decode(x, keep_k=keep_k)
        return {'points': points, 'kl': kl}


class OrderPreservingExplicitKLAutoEncoder(KLMixin, OrderPreservingExplicitAutoEncoder):
    """KL-regularized autoencoder that preserves input-output order correspondence.

    Inherits encoding logic from OrderPreservingExplicitAutoEncoder and adds KL
    regularization via the KLMixin.
    """

    def __init__(self, *, latent_dim=64, **kwargs):
        super().__init__(**kwargs)
        self._init_kl_components(kwargs.get('dim', 512), latent_dim)

    def encode(self, pc, num_faces=None):
        x = OrderPreservingExplicitAutoEncoder.encode(self, pc, num_faces=num_faces)
        return self._apply_kl_transform(x)

    def forward(self, pc, num_faces=None, keep_k=None):
        kl, x = self.encode(pc, num_faces=num_faces)
        points = self.decode(x, keep_k=keep_k)
        return {'points': points, 'kl': kl}


def create_autoencoder(dim=512, M=512, latent_dim=64, N=2048, determinisitc=False, point_dim=3, use_point_embed=True, use_nested_dropout=False, nested_dropout_size_sampling="uniform", use_causal_attention=False, use_legacy_attention=False, n_decoder_self_attn_blocks=24, decoder_use_rope=False):
    if determinisitc:
        model = AutoEncoder(
            decoder_depth=n_decoder_self_attn_blocks,
            dim=dim,
            queries_dim=dim,
            output_dim=1,
            num_inputs=N,
            num_latents=M,
            heads=8,
            dim_head=64,
            point_dim=point_dim,
            use_point_embed=use_point_embed,
            use_nested_dropout=use_nested_dropout,
            nested_dropout_size_sampling=nested_dropout_size_sampling,
            use_causal_attention=use_causal_attention,
            use_legacy_attention=use_legacy_attention,
            decoder_use_rope=decoder_use_rope,
        )
    else:
        model = KLAutoEncoder(
            decoder_depth=n_decoder_self_attn_blocks,
            dim=dim,
            queries_dim=dim,
            output_dim=1,
            num_inputs=N,
            num_latents=M,
            latent_dim=latent_dim,
            heads=8,
            dim_head=64,
            point_dim=point_dim,
            use_point_embed=use_point_embed,
            use_nested_dropout=use_nested_dropout,
            nested_dropout_size_sampling=nested_dropout_size_sampling,
            use_causal_attention=use_causal_attention,
            use_legacy_attention=use_legacy_attention,
            decoder_use_rope=decoder_use_rope,
        )
    return model


def create_learnable_query_autoencoder(dim=512, M=512, latent_dim=64, N=2048, determinisitc=False, point_dim=3, use_point_embed=True, use_nested_dropout=False, nested_dropout_size_sampling="uniform", use_causal_attention=False, use_legacy_attention=False, n_decoder_self_attn_blocks=24, decoder_use_rope=False):
    if determinisitc:
        model = LearnableQueryAutoEncoder(
            decoder_depth=n_decoder_self_attn_blocks,
            dim=dim,
            queries_dim=dim,
            output_dim=1,
            num_inputs=N,
            num_latents=M,
            heads=8,
            dim_head=64,
            point_dim=point_dim,
            use_point_embed=use_point_embed,
            use_nested_dropout=use_nested_dropout,
            nested_dropout_size_sampling=nested_dropout_size_sampling,
            use_causal_attention=use_causal_attention,
            use_legacy_attention=use_legacy_attention,
            decoder_use_rope=decoder_use_rope,
        )
    else:
        model = LearnableQueryKLAutoEncoder(
            decoder_depth=n_decoder_self_attn_blocks,
            dim=dim,
            queries_dim=dim,
            output_dim=1,
            num_inputs=N,
            num_latents=M,
            latent_dim=latent_dim,
            heads=8,
            dim_head=64,
            point_dim=point_dim,
            use_point_embed=use_point_embed,
            use_nested_dropout=use_nested_dropout,
            nested_dropout_size_sampling=nested_dropout_size_sampling,
            use_causal_attention=use_causal_attention,
            use_legacy_attention=use_legacy_attention,
            decoder_use_rope=decoder_use_rope,
        )
    return model


def create_explicit_autoencoder(
        dim=512,
        M=512,
        num_output_points=512,
        latent_dim=64,
        N=2048,
        determinisitc=False,
        point_dim=3,
        use_point_embed=True,
        use_nested_dropout=False,
        nested_dropout_size_sampling="uniform",
        use_causal_attention=False,
        use_legacy_attention=False,
        n_decoder_self_attn_blocks=24,
        n_encoder_self_attn_blocks=0,
        decoder_use_rope=False,
        encoder_use_rope=False,
        encoder_use_pos_embed=False,
        varlen_mode="nested_dropout",
        use_rank_pe_latents=False,
        num_sink_tokens=0,
):
    if determinisitc:
        model = LearnableQueryExplicitAutoEncoder(
            decoder_depth=n_decoder_self_attn_blocks,
            dim=dim,
            queries_dim=dim,
            num_inputs=N,
            num_latents=M,
            num_output_points=num_output_points,
            heads=8,
            dim_head=64,
            point_dim=point_dim,
            use_point_embed=use_point_embed,
            use_nested_dropout=use_nested_dropout,
            nested_dropout_size_sampling=nested_dropout_size_sampling,
            use_causal_attention=use_causal_attention,
            use_legacy_attention=use_legacy_attention,
            decoder_use_rope=decoder_use_rope,
            encoder_depth=n_encoder_self_attn_blocks,
            encoder_use_rope=encoder_use_rope,
            encoder_use_pos_embed=encoder_use_pos_embed,
            varlen_mode=varlen_mode,
            use_rank_pe_latents=use_rank_pe_latents,
            num_sink_tokens=num_sink_tokens,
        )
    else:
        model = LearnableQueryExplicitKLAutoEncoder(
            decoder_depth=n_decoder_self_attn_blocks,
            dim=dim,
            queries_dim=dim,
            num_inputs=N,
            num_latents=M,
            num_output_points=num_output_points,
            latent_dim=latent_dim,
            heads=8,
            dim_head=64,
            point_dim=point_dim,
            use_point_embed=use_point_embed,
            use_nested_dropout=use_nested_dropout,
            nested_dropout_size_sampling=nested_dropout_size_sampling,
            use_causal_attention=use_causal_attention,
            use_legacy_attention=use_legacy_attention,
            decoder_use_rope=decoder_use_rope,
            encoder_depth=n_encoder_self_attn_blocks,
            encoder_use_rope=encoder_use_rope,
            encoder_use_pos_embed=encoder_use_pos_embed,
            varlen_mode=varlen_mode,
            use_rank_pe_latents=use_rank_pe_latents,
            num_sink_tokens=num_sink_tokens,
        )
    return model


def create_order_preserving_explicit_autoencoder(
        dim=512,
        M=512,
        num_output_points=512,
        latent_dim=64,
        N=2048,
        determinisitc=False,
        point_dim=3,
        use_point_embed=True,
        use_nested_dropout=False,
        nested_dropout_size_sampling="uniform",
        use_causal_attention=False,
        use_legacy_attention=False,
        n_decoder_self_attn_blocks=24,
        n_encoder_self_attn_blocks=0,
        decoder_use_rope=False,
        encoder_use_rope=False,
        encoder_use_pos_embed=True,
        varlen_mode="nested_dropout",
        use_rank_pe_latents=False,
        num_sink_tokens=0,
        ae_cross_heads=1,
        query_binding="none",
        latent_quant_bits=0,
        latent_quant_train=False,
):
    """Create an order-preserving explicit autoencoder.

    This autoencoder uses fixed sinusoidal positional encoding as decoder queries
    to ensure that output position i corresponds to input position i.
    Encoder positional encoding is controlled by encoder_use_pos_embed.
    """
    if determinisitc:
        model = OrderPreservingExplicitAutoEncoder(
            decoder_depth=n_decoder_self_attn_blocks,
            dim=dim,
            queries_dim=dim,
            num_inputs=N,
            num_latents=M,
            num_output_points=num_output_points,
            heads=8,
            dim_head=64,
            point_dim=point_dim,
            use_point_embed=use_point_embed,
            use_nested_dropout=use_nested_dropout,
            nested_dropout_size_sampling=nested_dropout_size_sampling,
            use_causal_attention=use_causal_attention,
            use_legacy_attention=use_legacy_attention,
            decoder_use_rope=decoder_use_rope,
            encoder_depth=n_encoder_self_attn_blocks,
            encoder_use_rope=encoder_use_rope,
            encoder_use_pos_embed=encoder_use_pos_embed,
            varlen_mode=varlen_mode,
            use_rank_pe_latents=use_rank_pe_latents,
            num_sink_tokens=num_sink_tokens,
            ae_cross_heads=ae_cross_heads,
            query_binding=query_binding,
            latent_quant_bits=latent_quant_bits,
            latent_quant_train=latent_quant_train,
        )
    else:
        model = OrderPreservingExplicitKLAutoEncoder(
            decoder_depth=n_decoder_self_attn_blocks,
            dim=dim,
            queries_dim=dim,
            num_inputs=N,
            num_latents=M,
            num_output_points=num_output_points,
            latent_dim=latent_dim,
            heads=8,
            dim_head=64,
            point_dim=point_dim,
            use_point_embed=use_point_embed,
            use_nested_dropout=use_nested_dropout,
            nested_dropout_size_sampling=nested_dropout_size_sampling,
            use_causal_attention=use_causal_attention,
            use_legacy_attention=use_legacy_attention,
            decoder_use_rope=decoder_use_rope,
            encoder_depth=n_encoder_self_attn_blocks,
            encoder_use_rope=encoder_use_rope,
            encoder_use_pos_embed=encoder_use_pos_embed,
            varlen_mode=varlen_mode,
            use_rank_pe_latents=use_rank_pe_latents,
            num_sink_tokens=num_sink_tokens,
        )
    return model


def create_flow_explicit_autoencoder(
        dim=512,
        M=512,
        num_output_points=512,
        latent_dim=64,           # unused (no KL variant); kept for signature parity
        N=2048,
        determinisitc=True,      # flow AE is always deterministic-bottleneck
        point_dim=3,
        use_point_embed=False,
        use_nested_dropout=False,
        nested_dropout_size_sampling="uniform",
        use_causal_attention=False,
        use_legacy_attention=False,
        n_decoder_self_attn_blocks=8,
        n_encoder_self_attn_blocks=0,
        decoder_use_rope=False,
        encoder_use_rope=False,
        encoder_use_pos_embed=True,
        varlen_mode="nested_dropout",
        use_rank_pe_latents=False,
        num_sink_tokens=0,
        ae_cross_heads=1,
        flow_decoder_depth=8,
        flow_steps=20,
        flow_t_sampling="logit_normal",
        flow_conditioning="cross_attn",
        flow_mode_scale=0.25,
        flow_null_prob=0.0,
        flow_guidance=1.0,
        repa_weight=0.0,
        repa_layer=2,
        query_binding="none",
):
    """Create the flow-matching (generative-decoder) explicit autoencoder.

    Signature mirrors create_order_preserving_explicit_autoencoder so
    train_vocabulary can swap factories; n_decoder_self_attn_blocks sets the
    latent-conditioning self-attn depth, flow_decoder_depth the DiT blocks.
    """
    return OrderPreservingFlowExplicitAutoEncoder(
        decoder_depth=n_decoder_self_attn_blocks,
        query_binding=query_binding,
        flow_decoder_depth=flow_decoder_depth,
        flow_steps=flow_steps,
        flow_t_sampling=flow_t_sampling,
        flow_conditioning=flow_conditioning,
        flow_mode_scale=flow_mode_scale,
        flow_null_prob=flow_null_prob,
        flow_guidance=flow_guidance,
        repa_weight=repa_weight,
        repa_layer=repa_layer,
        dim=dim,
        queries_dim=dim,
        num_inputs=N,
        num_latents=M,
        num_output_points=num_output_points,
        heads=8,
        dim_head=64,
        point_dim=point_dim,
        use_point_embed=use_point_embed,
        use_nested_dropout=use_nested_dropout,
        nested_dropout_size_sampling=nested_dropout_size_sampling,
        use_causal_attention=use_causal_attention,
        use_legacy_attention=use_legacy_attention,
        decoder_use_rope=decoder_use_rope,
        encoder_depth=n_encoder_self_attn_blocks,
        encoder_use_rope=encoder_use_rope,
        encoder_use_pos_embed=encoder_use_pos_embed,
        varlen_mode=varlen_mode,
        use_rank_pe_latents=use_rank_pe_latents,
        num_sink_tokens=num_sink_tokens,
        ae_cross_heads=ae_cross_heads,
    )


def kl_d512_m512_l512(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=512, N=N, determinisitc=False)


def kl_d512_m512_l64(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=64, N=N, determinisitc=False)


def kl_d512_m512_l32(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=32, N=N, determinisitc=False)


def kl_d512_m512_l16(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=16, N=N, determinisitc=False)


def kl_d512_m512_l8(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=8, N=N, determinisitc=False)


def kl_d512_m512_l4(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=4, N=N, determinisitc=False)


def kl_d512_m512_l2(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=2, N=N, determinisitc=False)


def kl_d512_m512_l1(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=1, N=N, determinisitc=False)


###
def ae_d512_m512(N=2048):
    return create_autoencoder(dim=512, M=512, N=N, determinisitc=True)


def ae_d512_m256(N=2048):
    return create_autoencoder(dim=512, M=256, N=N, determinisitc=True)


def ae_d512_m128(N=2048):
    return create_autoencoder(dim=512, M=128, N=N, determinisitc=True)


def ae_d512_m64(N=2048):
    return create_autoencoder(dim=512, M=64, N=N, determinisitc=True)


###
def ae_d256_m512(N=2048):
    return create_autoencoder(dim=256, M=512, N=N, determinisitc=True)


def ae_d128_m512(N=2048):
    return create_autoencoder(dim=128, M=512, N=N, determinisitc=True)


def ae_d64_m512(N=2048):
    return create_autoencoder(dim=64, M=512, N=N, determinisitc=True)


# Learnable query version example
def lq_kl_d512_m512_l64(N=2048):
    return create_learnable_query_autoencoder(dim=512, M=512, latent_dim=64, N=N, determinisitc=False)


# Explicit generation example
def exp_kl_d512_m512_out512_l64(N=2048):
    return create_explicit_autoencoder(dim=512, M=512, num_output_points=512, latent_dim=64, N=N, determinisitc=False)
