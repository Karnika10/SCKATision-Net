from __future__ import annotations

import math
from dataclasses import asdict

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig


class GroupBatchNorm2d(nn.Module):
    def __init__(self, channels: int, groups: int = 16, eps: float = 1e-5):
        super().__init__()
        groups = min(groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.gn = nn.GroupNorm(groups, channels, eps=eps, affine=True)

    @property
    def weight(self):
        return self.gn.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x)


class SpatialReconstructionUnit(nn.Module):
    """SRU block inspired by SC-Conv separation-reconstruction gating."""

    def __init__(self, channels: int, groups: int = 16, gate_threshold: float = 0.5):
        super().__init__()
        self.norm = GroupBatchNorm2d(channels, groups)
        self.gate_threshold = gate_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        gamma = self.norm.weight / (self.norm.weight.sum().clamp_min(1e-6))
        gamma = gamma.view(1, -1, 1, 1)
        gates = torch.sigmoid(x_norm * gamma)
        informative = gates >= self.gate_threshold
        x_info = informative * x
        x_non = (~informative) * x
        c = x.shape[1]
        if c % 2 != 0:
            return x_info + x_non
        x1, x2 = torch.chunk(x_info, 2, dim=1)
        y1, y2 = torch.chunk(x_non, 2, dim=1)
        return torch.cat([x1 + y2, x2 + y1], dim=1)


class ChannelReconstructionUnit(nn.Module):
    """CRU split-transform-fuse channel refinement."""

    def __init__(self, channels: int, alpha: float = 0.5, squeeze_ratio: int = 2, group_size: int = 2):
        super().__init__()
        upper = int(channels * alpha)
        lower = channels - upper
        upper_sq = max(1, upper // squeeze_ratio)
        lower_sq = max(1, lower // squeeze_ratio)
        groups = max(1, min(group_size, upper_sq))
        while upper_sq % groups != 0 and groups > 1:
            groups -= 1
        self.upper = upper
        self.lower = lower
        self.squeeze_u = nn.Conv2d(upper, upper_sq, 1, bias=False)
        self.squeeze_l = nn.Conv2d(lower, lower_sq, 1, bias=False)
        self.gwc = nn.Conv2d(upper_sq, channels, 3, padding=1, groups=groups, bias=False)
        self.pwc_u = nn.Conv2d(upper_sq, channels, 1, bias=False)
        self.pwc_l = nn.Conv2d(lower_sq, channels, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xu, xl = torch.split(x, [self.upper, self.lower], dim=1)
        xu = self.squeeze_u(xu)
        xl = self.squeeze_l(xl)
        yu = self.gwc(xu) + self.pwc_u(xu)
        yl = self.pwc_l(xl)
        y = torch.cat([yu, yl], dim=1)
        weights = F.softmax(self.pool(y), dim=1)
        y = y * weights
        y1, y2 = torch.chunk(y, 2, dim=1)
        return y1 + y2


class SCConvEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.sru = SpatialReconstructionUnit(out_channels)
        self.cru = ChannelReconstructionUnit(out_channels)
        self.out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.sru(x)
        x = self.cru(x)
        return self.out(x)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self.proj(x)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return x, (h, w)


class GlobalLocalAttentionEncoder(nn.Module):
    """Combines local depth-wise convolution over patch tokens with global MHA."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        cls, patches = x[:, :1], x[:, 1:]
        b, n, c = patches.shape
        h, w = hw
        patches_2d = patches.transpose(1, 2).reshape(b, c, h, w)
        local = self.local(patches_2d).flatten(2).transpose(1, 2)
        x_local = torch.cat([cls, local], dim=1)
        x_norm = self.norm(x_local)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        return x + self.dropout(attn_out)


class KANLayer(nn.Module):
    """Practical RBF KAN-style layer with learnable edge functions.

    It approximates KAN edge activations by applying a learnable linear projection over
    Gaussian radial basis expansions, followed by a residual linear shortcut.
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        centers = torch.linspace(-1.0, 1.0, grid_size)
        self.register_buffer("centers", centers)
        self.log_width = nn.Parameter(torch.zeros(1))
        self.rbf_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size))
        self.base = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.rbf_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clip = torch.tanh(x)
        width = F.softplus(self.log_width) + 1e-3
        basis = torch.exp(-((x_clip.unsqueeze(-1) - self.centers) / width) ** 2)
        out = torch.einsum("...ig,oig->...o", basis, self.rbf_weight)
        return out + self.base(x) + self.bias


class KANFeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1, grid_size: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            KANLayer(embed_dim, hidden_dim, grid_size),
            nn.GELU(),
            nn.Dropout(dropout),
            KANLayer(hidden_dim, embed_dim, grid_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SCKATisionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, kan_hidden_dim: int, dropout: float, grid_size: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.kan = KANFeedForward(embed_dim, kan_hidden_dim, dropout, grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop1(y)
        x = x + self.kan(self.norm2(x))
        return x


class SCKATisionNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        stem_channels = max(32, config.embed_dim // 2)
        self.scconv = SCConvEncoder(config.in_channels, stem_channels)
        self.patch_embed = PatchEmbedding(stem_channels, config.embed_dim, config.patch_size)
        num_patches = (config.image_size // config.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.pos_drop = nn.Dropout(config.dropout)
        self.glae = GlobalLocalAttentionEncoder(config.embed_dim, config.num_heads, config.dropout)
        self.blocks = nn.ModuleList([
            SCKATisionBlock(config.embed_dim, config.num_heads, config.kan_hidden_dim, config.dropout, config.rbf_grid_size)
            for _ in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.scconv(x)
        x, hw = self.patch_embed(x)
        b = x.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if x.shape[1] != self.pos_embed.shape[1]:
            # Allows non-default image sizes by interpolating positional embeddings.
            pos = self._interpolate_pos_embed(hw)
        else:
            pos = self.pos_embed
        x = self.pos_drop(x + pos)
        x = self.glae(x, hw)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]

    def _interpolate_pos_embed(self, hw: tuple[int, int]) -> torch.Tensor:
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        old = int(math.sqrt(patch_pos.shape[1]))
        patch_pos = patch_pos.reshape(1, old, old, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=hw, mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, hw[0] * hw[1], -1)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_model(**kwargs) -> SCKATisionNet:
    config = ModelConfig(**kwargs)
    return SCKATisionNet(config)
