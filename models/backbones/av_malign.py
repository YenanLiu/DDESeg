# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
from einops import rearrange


def norm_fea(fea, norm_flag):
    if norm_flag:
        return F.normalize(fea, p=2, dim=-1)
    return fea

class Cross_modal_interaction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, img_fea, aud_fea, aq=True, sigmoid=False, softmax=False):
        """
        Args:
            img_fea: [b, num_token, dim]
            aud_fea: [b, k, dim]
            aq: True: a is query; False: i is query
        Return
            if aq:
                fuse_fea: [b, k, dim]
            else:
                fuse_fea: [b, num_token, dim]
        """
        if len(aud_fea.shape) == 2:
            aud_fea = aud_fea.unsqueeze(1)
        if aq:
            B, N, C = aud_fea.shape
            query = aud_fea
            key = img_fea
            value = img_fea
        else:
            B, N, C = img_fea.shape # [B, h*w, dim]
            query = img_fea
            key = aud_fea
            value = aud_fea
        
        q = rearrange(self.q_proj(query), 'b n (h c) -> b h n c', h=self.num_heads)  # [B, nheads, h*w, dim//nheads] 
        k = rearrange(self.k_proj(key), 'b n (h c) -> b h n c', h=self.num_heads) # [B, nheads, 1, dim//nheads] 
        v = rearrange(self.v_proj(value), 'b n (h c) -> b h n c', h=self.num_heads)  # [B, nheads, 1, dim//nheads] 
        
        attn = (q @ k.transpose(-2, -1)) * self.scale 
        if sigmoid:
            attn = torch.sigmoid(attn) # [B, nheads, h*w, 1] 
        if softmax:
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 

        return x
    
class GumbelSoftmaxClustering(nn.Module):
    def __init__(self, num_clusters, dim, temperature=1.0):
        super(GumbelSoftmaxClustering, self).__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, dim))  # Learnable cluster centers

    def forward(self, img_fea):
        """
        Args:
            img_fea: [B, H*W, dim] -> image features
        Returns:
            cluster_assignments: [B, H*W, num_clusters] -> soft cluster assignments
            cluster_centers: [B, H*W, dim] -> recomputed cluster centers
        """
        # Compute similarity (dot product) between img_fea and cluster centers
        logits = torch.einsum('bnd,kd->bnk', img_fea, self.cluster_centers)  # [B, H*W, num_clusters]

        # Apply Gumbel Softmax for cluster assignment
        gumbel_samples = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)  # [B, H*W, num_clusters]

        cluster_centers = torch.einsum('bnk,kd->bnd', gumbel_samples, self.cluster_centers)  # [B, H*W, dim]
        # cluster_centers = torch.einsum('bnk,bnd->bkd', gumbel_samples, img_fea) # 

        return cluster_centers, gumbel_samples
    
class AV_Aligner(nn.Module):
    def __init__(self, dim, num_heads=1, num_clusters=5):
        super(AV_Aligner, self).__init__()

        self.num_clusters = num_clusters

        self.fc = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(dim)

        self.gumbel_clustering = GumbelSoftmaxClustering(num_clusters, dim)

        self.av_cros_interaction = Cross_modal_interaction(dim=dim, num_heads=num_heads)
        self.va_cros_interaction = Cross_modal_interaction(dim=dim, num_heads=num_heads)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, aud_fea, img_fea):
        """
        Args:
            aud_fea: [B, K, dim] -> audio features
            img_fea: [B, H*W, dim] -> image features
        Returns:
            enhanced_img_fea: [B, H*W, dim] -> enhanced image features
        """
        # Step 1: Cluster image features into K clusters
        cluster_centers, _ = self.gumbel_clustering(img_fea) # [B, img_cluster, dim]

        aud_enhance_fea = self.av_cros_interaction(cluster_centers, aud_fea, aq=True, softmax=True) # [B, k, dim]
        attention_scores = self.fc(aud_enhance_fea)  # [B, k, 1] 
        attention_scores = attention_scores.softmax(dim=1)  #[B, k, 1]

        weighted_aud_fea = attention_scores * aud_fea  # [B, k, 768]
        weighted_aud_fea = weighted_aud_fea.sum(dim=1) # [B, 768]

        # Step 6: Enhance image features based on the audio-visual similarity map
        enhanced_img_fea = self.va_cros_interaction(img_fea, weighted_aud_fea, aq=False, sigmoid=True)  

        # enhanced_img_fea = self.layer_norm(enhanced_img_fea + img_fea)
        enhanced_img_fea = enhanced_img_fea + img_fea 

        return enhanced_img_fea