"""
Vision Transformer with LRP capabilities for RETFound

This module extends the original RETFound VisionTransformer model with 
Layer-wise Relevance Propagation (LRP) capabilities for explainability.

Adapted from:
- RETFound (https://github.com/rmaphoh/RETFound_MAE/)
- Transformer-Explainability (https://github.com/hila-chefer/Transformer-Explainability/)
"""

from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer
from einops import rearrange
import sys
import os

# Set path to Transformer-Explainability - update this path as needed
BASELINE_PATH = "/HDD/data/yating/ocular-llm-explainability/Transformer-Explainability"
if BASELINE_PATH not in sys.path:
    sys.path.insert(0, BASELINE_PATH)

# Import LRP-compatible layers
from modules.layers_ours import (
    IndexSelect, Linear, LayerNorm, Dropout, GELU, 
    Softmax, Conv2d, einsum, Clone, Add
)

# Weight initialization function
from baselines.ViT.weight_init import trunc_normal_
from baselines.ViT.layer_helpers import to_2tuple

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    """
    Compute rollout attention across layers
    """
    # Adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    
    # Optional normalization
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                      for i in range(len(all_layer_matrices))]
    
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)

        # Storage for attention maps and gradients
        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        self.save_v(v)

        dots = self.matmul1([q, k]) * self.scale

        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        self.save_attn(attn)
        
        attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # attn = A*V
        (cam1, cam_v) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x):
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(1, 2)
        cam = cam.reshape(cam.shape[0], cam.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        return self.proj.relprop(cam, **kwargs)


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    Vision Transformer with support for global average pooling and LRP
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                num_heads=12, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                global_pool=True, mlp_head=False):
        super().__init__(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
        )
        
        # Override the default RETFound configuration
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  
        self.global_pool = global_pool
        
        # Replace standard layers with LRP-compatible ones
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        # add from Transformer-Explainability
        num_patches = self.patch_embed.num_patches

        # add from Transformer-Explainability
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Create LRP-compatible blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate, drop=drop_rate)
            for _ in range(depth)])
        
        # Apply correct normalization for global pooling
        if global_pool:
            self.fc_norm = LayerNorm(embed_dim)
            if hasattr(self, 'norm'):
                del self.norm  # remove the original norm
        else:
            self.norm = LayerNorm(embed_dim)
        
        # Create proper head
        if mlp_head:
            self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
        else:
            self.head = Linear(embed_dim, num_classes)
             
        # Proper weight init
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # LRP tools
        self.pool = IndexSelect()
        self.add = Add()
        
        # For storing gradients
        self.inp_grad = None

    def save_inp_grad(self, grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        
"""
Vision Transformer with LRP capabilities for RETFound

This module extends the original RETFound VisionTransformer model with 
Layer-wise Relevance Propagation (LRP) capabilities for explainability.

Adapted from:
- RETFound (https://github.com/rmaphoh/RETFound_MAE/)
- Transformer-Explainability (https://github.com/hila-chefer/Transformer-Explainability/)
"""

from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer
from einops import rearrange
import sys
import os

# Set path to Transformer-Explainability - update this path as needed
BASELINE_PATH = "/HDD/data/yating/ocular-llm-explainability/Transformer-Explainability"
if BASELINE_PATH not in sys.path:
    sys.path.insert(0, BASELINE_PATH)

# Import LRP-compatible layers
from modules.layers_ours import (
    IndexSelect, Linear, LayerNorm, Dropout, GELU, 
    Softmax, Conv2d, einsum, Clone, Add
)

# Weight initialization function
from baselines.ViT.weight_init import trunc_normal_
from baselines.ViT.layer_helpers import to_2tuple

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    """
    Compute rollout attention across layers
    """
    # Adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    
    # Optional normalization
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                      for i in range(len(all_layer_matrices))]
    
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)

        # Storage for attention maps and gradients
        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        self.save_v(v)

        dots = self.matmul1([q, k]) * self.scale

        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        self.save_attn(attn)
        attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # attn = A*V
        (cam1, cam_v) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x):
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(1, 2)
        cam = cam.reshape(cam.shape[0], cam.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        return self.proj.relprop(cam, **kwargs)


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    Vision Transformer with support for global average pooling and LRP
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                num_heads=12, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                global_pool=True, mlp_head=False):
        super().__init__(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
        )
        
        # Override the default RETFound configuration
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  
        self.global_pool = global_pool
        
        # Replace standard layers with LRP-compatible ones
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        # add from Transformer-Explainability
        num_patches = self.patch_embed.num_patches

        # add from Transformer-Explainability
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Create LRP-compatible blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate, drop=drop_rate)
            for _ in range(depth)])
        
        # Apply correct normalization for global pooling
        if global_pool:
            self.fc_norm = LayerNorm(embed_dim)
            if hasattr(self, 'norm'):
                del self.norm  # remove the original norm
        else:
            self.norm = LayerNorm(embed_dim)
        
        # Create proper head
        if mlp_head:
            self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
        else:
            self.head = Linear(embed_dim, num_classes)
             
        # Proper weight init
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # LRP tools
        self.pool = IndexSelect()
        self.add = Add()
        
        # For storing gradients
        self.inp_grad = None

    def save_inp_grad(self, grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        # Enable gradient tracking for attribution
        x.requires_grad_(True)
        x.register_hook(self.save_inp_grad)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            # RETFound global pooling: exclude CLS token, average all other tokens
            tokens = x[:, 1:, :]  # All tokens except CLS
            x = tokens.mean(dim=1, keepdim=True)  # Global pool           
            x = self.fc_norm(x)
            x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
            x = x.squeeze(1)  # Remove the keepdim
            x = self.head(x)
        else:
            # Use CLS token path
            x = self.norm(x)
            x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
            x = x.squeeze(1)
            x = self.head(x)

        return x

    def relprop(self, cam=None, method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        """
        Relevance propagation for the Vision Transformer
        
        Args:
            cam: Starting relevance scores (usually one-hot encoded vector for classification)
            method: Attribution method to use
            is_ablation: Whether to use ablation study mode
            start_layer: First layer to consider for rollout computation
            
        Returns:
            Relevance scores for the input patches
        """
        # Propagate through the head
        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        # Handle different pooling strategies  ->  This is different from the Transformer-Explainability repo, because RETFound ViT has specific global pooling with fc_norm
        if self.global_pool:  
            # Apply fc_norm - Note: this won't backpropagate to the CLS token
            #cam = self.pool.relprop(cam, **kwargs)
            cam = self.pool.relprop(cam, **kwargs)
            cam = self.fc_norm.relprop(cam, **kwargs)                    
            # Expand relevance to all tokens (including CLS) for proper backprop
            # This is needed because we pooled all non-CLS tokens, so relevance
            # must be distributed back to all of them
            
            
            expanded_cam = torch.zeros((cam.shape[0], 
                               self.patch_embed.num_patches + 1, 
                               cam.shape[2]), 
                              device=cam.device)    
            
            # Add: Set CLS token relevance to zero since it's not used in global pooling
            expanded_cam[:, 0, :] = 0
            
            expanded_cam[:, 1:, :] = cam.repeat(1, self.patch_embed.num_patches, 1)
            cam = expanded_cam   
        else:
            cam = self.pool.relprop(cam, **kwargs)
            cam = self.norm.relprop(cam, **kwargs)
        
        # Propagate through transformer blocks
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)
            
        # Propagate through position embedding addition
        # (cam, _) = self.add.relprop(cam, **kwargs)  # remove because the Transformer-Explainability repo does not have this
            
        # For transformer_attribution method (our method), use attention rollout
        if method == "transformer_attribution" or method == "grad":
            # Collect attention cams from all blocks
            cams = []
            for blk in self.blocks:
                # Get gradients and attention maps
                grad = blk.attn.get_attn_gradients()
                cam_layer = blk.attn.get_attn_cam()
                
                # Reshape for processing
                cam_layer = cam_layer[0].reshape(-1, cam_layer.shape[-1], cam_layer.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                
                # Element-wise product and mean across heads
                cam = grad * cam_layer
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))

            # Add: check if we have valid cams before proceeding
            if len(cams) == 0:
                print("Warning: No valid attribution maps generated. Check gradient flow.")
                # Return a fallback attribution (uniform)
                return torch.ones(1, self.patch_embed.num_patches, device=cam.device) / self.patch_embed.num_patches
                
            # Compute rollout attention across all layers
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            
            # Extract relevance for non-CLS tokens
            # Relevance flows from CLS (0) to patches (1:)
            if self.global_pool:
            # For global pooling, use average influence from all tokens to patches
            # This avoids the horizontal lines artifact
                cam = rollout[:, 1:, 1:].mean(dim=1)
            else:
            # For CLS token, use standard approach
                cam = rollout[:, 0, 1:]
            return cam
        
        # Full method - project relevance all the way to pixel space
        elif method == "full":
            # Get only patch relevance (exclude CLS token)
            cam = cam[:, 1:]         
            # Project back to pixel space with patch_embed
            cam = self.patch_embed.relprop(cam, **kwargs)        
            # Sum across channels
            cam = cam.sum(dim=1)
            return cam
        
        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            rollout = compute_rollout_attention(attn_cams, start_layer=start_layer)
            if self.global_pool:
            # For global pooling, use average influence from all tokens to patches
            # This avoids the horizontal lines artifact
                cam = rollout[:, 1:, 1:].mean(dim=1)
            else:
            # For CLS token, use standard approach
                cam = rollout[:, 0, 1:]
            return cam
        

def RETFound_mae(**kwargs):
    """
    Create RETFound model with LRP capabilities
    """
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, **kwargs)
    
    return model

def RETFound_dinov2(args, **kwargs):
    """
    Create RETFound model based on DINO-v2 with LRP capabilities
    """
    # Note: DINOv2 models would need additional LRP implementation
    # since they are loaded from timm directly
    model = timm.create_model(
        'vit_large_patch14_dinov2.lvd142m',
        pretrained=True,
        img_size=224,
        **kwargs
    )
    return model