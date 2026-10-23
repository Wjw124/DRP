"""Atomic-action MAE and motion aggregation token. Input: [B,T,K,P,P,2]."""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class PatchConvEmbed(nn.Module):
    """Embed each flow patch with a convolution."""

    def __init__(self, in_chans=2, patch_size=16, embed_dim=64):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=1)
        self.patch_size = patch_size

    def forward(self, patches):
        (B, T, N_p, H_p, W_p, C) = patches.shape
        patches = patches.view(B * T * N_p, C, H_p, W_p)
        x = self.conv(patches)
        x = x.view(B, T, N_p, -1)
        return x


class PatchConvRecover(nn.Module):
    """Recover flow patches from their embeddings."""

    def __init__(self, embed_dim=64, out_chans=2, patch_size=16):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            embed_dim, out_chans, kernel_size=patch_size, stride=1
        )
        self.patch_size = patch_size

    def forward(self, x):
        (B, T, N_p, D) = x.shape
        x = x.view(B * T * N_p, D, 1, 1)
        patches = self.deconv(x)
        patches = patches.view(B, T, N_p, self.patch_size, self.patch_size, -1)
        return patches


class PositionalEncoding(nn.Module):
    """Learned position embeddings with optional indexed lookup."""

    def __init__(self, max_num_patches, embed_dim):
        super().__init__()
        self.max_num_patches = max_num_patches
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, max_num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, indices=None):
        (B, N, D) = x.shape
        if indices is None:
            pos_embed = self.pos_embed[:, :N, :]
        else:
            pos_embed = self.pos_embed.expand(B, self.max_num_patches, -1)
            batch_indices = torch.arange(B, device=x.device)[:, None]
            pos_embed = pos_embed[batch_indices, indices]
        return x + pos_embed


class IntraFrameAttention(nn.Module):
    """Attention among patch tokens within each frame."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        (B, T, N_p, D) = x.shape
        qkv = self.qkv(x).reshape(B, T, N_p, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)
        (q, k, v) = (qkv[0], qkv[1], qkv[2])
        attn = q @ k.transpose(-2, -1) * self.head_dim ** (-0.5)
        if mask is not None:
            mask = mask.unsqueeze(2).unsqueeze(2)
            attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(2, 3).contiguous()
        x = x.reshape(B, T, N_p, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class InterFrameAttention(nn.Module):
    """Causal attention along each tracked point over time."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        (B, T, N_p, D) = x.shape
        x = x.permute(0, 2, 1, 3)
        qkv = self.qkv(x).reshape(B, N_p, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)
        (q, k, v) = (qkv[0], qkv[1], qkv[2])
        attn = q @ k.transpose(-2, -1) * self.head_dim ** (-0.5)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(2, 3).contiguous()
        x = x.reshape(B, N_p, T, D)
        x = x.permute(0, 2, 1, 3)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DualAttentionBlock(nn.Module):
    """Apply spatial and temporal attention with residual connections."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.1,
        use_intra_attention=True,
        use_inter_attention=True,
    ):
        super().__init__()
        self.use_intra_attention = use_intra_attention
        self.use_inter_attention = use_inter_attention
        self.norm1 = nn.LayerNorm(embed_dim)
        if self.use_intra_attention:
            self.intra_attn = IntraFrameAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        if self.use_inter_attention:
            self.inter_attn = InterFrameAttention(embed_dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )
        self.standard_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x, intra_mask=None):
        if x.dim() == 4:
            if self.use_intra_attention:
                x = x + self.intra_attn(self.norm1(x), mask=intra_mask)
            if self.use_inter_attention:
                x = x + self.inter_attn(self.norm2(x))
            x = x + self.mlp(self.norm3(x))
        else:
            residual = x
            x = self.norm1(x)
            (x, _) = self.standard_attn(x, x, x)
            x = residual + x
            x = x + self.mlp(self.norm3(x))
        return x


class TransformerEncoder(nn.Module):
    """Stacked transformer encoder."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers,
        mlp_ratio=4.0,
        use_dual_attention=False,
        num_patches=16,
        use_intra_attention=True,
        use_inter_attention=True,
    ):
        super().__init__()
        self.use_dual_attention = use_dual_attention
        self.num_patches = num_patches
        if use_dual_attention:
            self.layers = nn.ModuleList(
                [
                    DualAttentionBlock(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        use_intra_attention=use_intra_attention,
                        use_inter_attention=use_inter_attention,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dim_feedforward=int(embed_dim * mlp_ratio),
                        activation="gelu",
                        batch_first=True,
                    )
                    for _ in range(num_layers)
                ]
            )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        if self.use_dual_attention and x.dim() == 4:
            for layer in self.layers:
                x = layer(x, intra_mask=mask)
        elif self.use_dual_attention and x.dim() == 3:
            B = x.shape[0]
            seq_len = x.shape[1]
            D = x.shape[2]
            N = self.num_patches + 1
            if seq_len % N == 0:
                T = seq_len // N
                x = x.view(B, T, N, D)
                for layer in self.layers:
                    x = layer(x, intra_mask=mask)
                x = x.view(B, T * N, D)
            else:
                for layer in self.layers:
                    x = layer(x)
        else:
            for layer in self.layers:
                x = layer(x)
        x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):
    """Self-attention decoder for masked patch reconstruction."""

    def __init__(self, embed_dim, num_heads, num_layers, mlp_ratio=4.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DualAttentionEncoder(nn.Module):
    """Stacked dual-attention encoder."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers,
        mlp_ratio=4.0,
        dropout=0.1,
        use_intra_attention=True,
        use_inter_attention=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DualAttentionBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    dropout,
                    use_intra_attention,
                    use_inter_attention,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, intra_mask=mask)
        return x


class PatchMAE(nn.Module):
    """Encode tracked flow patches and reconstruct masked patches."""

    def __init__(
        self,
        patch_size=16,
        embed_dim=32,
        num_heads=8,
        enc_layers=6,
        dec_layers=4,
        mlp_ratio=4.0,
        mask_ratio=0.25,
        aggregate_method="cls",
        num_patches=16,
        use_dual_attention=True,
        use_intra_attention=True,
        use_inter_attention=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.aggregate_method = aggregate_method
        self.num_patches = num_patches
        self.use_dual_attention = use_dual_attention
        self.use_intra_attention = use_intra_attention
        self.use_inter_attention = use_inter_attention
        self.patch_embed = PatchConvEmbed(2, patch_size, embed_dim)
        self.patch_recover = PatchConvRecover(embed_dim, 2, patch_size)
        self.max_time_steps = 50
        self.time_embed = nn.Parameter(torch.zeros(1, self.max_time_steps, embed_dim))
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        self.track_pos_encoder = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.encoder = TransformerEncoder(
            embed_dim,
            num_heads,
            enc_layers,
            mlp_ratio,
            use_dual_attention,
            num_patches,
            use_intra_attention,
            use_inter_attention,
        )
        self.decoder = TransformerDecoder(embed_dim, num_heads, dec_layers, mlp_ratio)
        self.head = nn.Linear(embed_dim, embed_dim)

    def forward(self, patches, tracks=None):
        (B, T, N_p, H_p, W_p, C) = patches.shape
        num_patches = N_p * T
        patch_tokens = self.patch_embed(patches)
        tracks_normalized = None
        track_pos_encoding = None
        if tracks is not None:
            tracks_normalized = tracks / 32.0 - 1.0
            track_pos_encoding = self.track_pos_encoder(tracks_normalized)
        cls_tokens = self.cls_token.expand(B, T, 1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=2)
        num_masked = int(N_p * self.mask_ratio)
        rand_indices = torch.rand(B, T, N_p, device=patches.device).argsort(dim=2)
        mask_indices = rand_indices[:, :, :num_masked]
        unmask_indices = rand_indices[:, :, num_masked:]
        if self.use_dual_attention:
            mask_map = torch.zeros(
                B, T, N_p + 1, dtype=torch.bool, device=patches.device
            )
            batch_indices = torch.arange(B, device=patches.device)[:, None, None]
            time_indices = torch.arange(T, device=patches.device)[None, :, None]
            mask_map[batch_indices, time_indices, mask_indices + 1] = True
            masked_tokens = tokens.clone()
            patch_mask_map = mask_map[:, :, 1:]
            num_masked_total = patch_mask_map.sum().item()
            masked_tokens[:, :, 1:, :][patch_mask_map] = self.mask_token.expand(
                num_masked_total, -1
            )
            if tracks is not None:
                masked_tokens[:, :, 1:, :] += track_pos_encoding
            time_embed_all = (
                self.time_embed[:, :T, :].unsqueeze(2).expand(B, T, N_p + 1, -1)
            )
            masked_tokens = masked_tokens + time_embed_all
            enc_output = self.encoder(masked_tokens)
            dec_input = enc_output.clone()
            if tracks is not None:
                dec_input[:, :, 1:, :] = dec_input[:, :, 1:, :] + track_pos_encoding
            time_embed_all = (
                self.time_embed[:, :T, :].unsqueeze(2).expand(B, T, N_p + 1, -1)
            )
            dec_input = dec_input + time_embed_all
            dec_input = dec_input.view(B, T * (N_p + 1), -1)
            dec_output = self.decoder(dec_input)
            dec_output = dec_output.view(B, T, N_p + 1, -1)
            patch_output = dec_output[:, :, 1:, :]
            patch_output = patch_output.contiguous().view(B, T, N_p, -1)
            pred_patches = self.head(patch_output)
        else:
            batch_indices = torch.arange(B, device=patches.device)[:, None, None]
            time_indices = torch.arange(T, device=patches.device)[None, :, None]
            select_mask = torch.ones(
                B, T, N_p + 1, dtype=torch.bool, device=patches.device
            )
            select_mask[batch_indices, time_indices, mask_indices + 1] = False
            masked_tokens = tokens.clone()
            patch_select_mask = select_mask[:, :, 1:]
            patch_mask_positions = ~patch_select_mask
            num_masked_total = patch_mask_positions.sum().item()
            masked_tokens[:, :, 1:, :][patch_mask_positions] = self.mask_token.expand(
                num_masked_total, -1
            )
            if tracks is not None:
                masked_tokens[:, :, 1:, :] += track_pos_encoding
            time_embed_all = (
                self.time_embed[:, :T, :].unsqueeze(2).expand(B, T, N_p + 1, -1)
            )
            masked_tokens = masked_tokens + time_embed_all
            masked_tokens = masked_tokens.view(B, T * (N_p + 1), -1)
            enc_output = self.encoder(masked_tokens)
            enc_output_4d = enc_output.view(B, T, N_p + 1, -1)
            dec_input = enc_output_4d.clone()
            if tracks is not None:
                dec_input[:, :, 1:, :] = dec_input[:, :, 1:, :] + track_pos_encoding
            time_embed_all = (
                self.time_embed[:, :T, :].unsqueeze(2).expand(B, T, N_p + 1, -1)
            )
            dec_input = dec_input + time_embed_all
            dec_input = dec_input.view(B, T * (N_p + 1), -1)
            dec_output = self.decoder(dec_input)
            patch_output = dec_output.view(B, T, N_p + 1, -1)[:, :, 1:, :]
            patch_output = patch_output.contiguous().view(B, T, N_p, -1)
            pred_patches = self.head(patch_output)
        recon_patches = self.patch_recover(pred_patches)
        with torch.no_grad():
            target = patches.detach()
        mask_map = torch.zeros(B, T, N_p, device=patches.device)
        batch_indices = torch.arange(B, device=patches.device)[:, None, None].expand(
            B, T, num_masked
        )
        time_indices = torch.arange(T, device=patches.device)[None, :, None].expand(
            B, T, num_masked
        )
        mask_map[batch_indices, time_indices, mask_indices] = 1
        recon_patches_flat = recon_patches.view(B, T, N_p, -1)
        target_flat = target.view(B, T, N_p, -1)
        mask_map_flat = mask_map.bool().view(B, T, N_p)
        recon_masked = recon_patches_flat[mask_map_flat]
        target_masked = target_flat[mask_map_flat]
        loss = F.mse_loss(recon_masked, target_masked, reduction="mean")
        return (recon_patches, loss)

    def encode(self, patches, mask_ratio=0.0, tracks=None):
        (B, T, N_p, H_p, W_p, C) = patches.shape
        patch_tokens = self.patch_embed(patches)
        track_pos_encoding = None
        if tracks is not None:
            tracks_normalized = tracks / 32.0 - 1.0
            track_pos_encoding = self.track_pos_encoder(tracks_normalized)
        cls_tokens = self.cls_token.expand(B, T, 1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=2)
        if self.use_dual_attention:
            if tracks is not None:
                tokens[:, :, 1:, :] += track_pos_encoding
            time_embed_all = (
                self.time_embed[:, :T, :].unsqueeze(2).expand(B, T, N_p + 1, -1)
            )
            tokens = tokens + time_embed_all
            enc_output = self.encoder(tokens)
            cls_features = enc_output[:, :, 0, :]
        else:
            if tracks is not None:
                tokens[:, :, 1:, :] += track_pos_encoding
            time_embed_all = (
                self.time_embed[:, :T, :].unsqueeze(2).expand(B, T, N_p + 1, -1)
            )
            tokens = tokens + time_embed_all
            tokens = tokens.view(B, T * (N_p + 1), -1)
            enc_output = self.encoder(tokens)
            enc_output = enc_output.view(B, T, N_p + 1, -1)
            cls_features = enc_output[:, :, 0, :]
        return cls_features

    def get_flow_token(self, patches, mask_ratio=0.0, tracks=None):
        return self.encode(patches, mask_ratio, tracks)
