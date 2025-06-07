
import math
import torch.nn as nn
import math

from .pre_modules import trunc_normal_, DropPath, Mlp

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape # [B, 128*128, dim]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # [B, 128*128, dim]->[B, 128*128, num_head, dim // num_heads]->[B, num_head, 128*128, dim // num_heads]

        if self.sr_ratio > 1:  
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W) # [B, dim, 128*128] -> [B, dim, 128, 128]
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1) # [B, dim, 16, 16] -> [B, dim, 16*16] -> [B, 16*16, dim]
            x_ = self.norm(x_) # [B, 16*16, dim]
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # [B, 16*16, dim]->[B, 16*16, 2, num_heads, dim//num_heads]->[2, B, num_heads, 16*16, dim//num_heads]
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # [B, 128*128, dim]->[B, 128*128, 2, num_heads, dim//num_heads]->[2, B, num_heads, 16*16, dim//num_heads]
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, num_heads, 128*128, dim//num_heads] * [B, num_heads, dim//num_heads, 16*16] -> [B, num_heads, 128*128, 16*16]
        attn = attn.softmax(dim=-1) # [B, num_heads, 128*128, 16*16]
        attn = self.attn_drop(attn) # [B, num_heads, 128*128, 16*16]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # [B, num_heads, 128*128, 16*16] * [B, num_heads, 16*16, dim//num_heads]->[B, num_heads, 128*128, dim//num_heads]->[B, 128*128, dim]
        x = self.proj(x) # [B, 128*128, dim]
        x = self.proj_drop(x) # [B, 128*128, dim]

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
                                dim,
                                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio
                            )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W)) # torch.Size([20, 16384, 32])
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W)) # torch.Size([20, 16384, 32])
        return x
    