""" 
Copy from pytorch-image-models to ease personalized modifications.
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch; torch.hub.set_dir('/2tssd/rzuo/pretrained_models/')
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
# from timm.models.helpers import adapt_input_conv
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


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

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


def gen_ip_mask(n_per_region, N, coord, patch_rela='two_step'):
    # coord [T,3,2], mask [T,N,1] or [T,1,N,N]
    T = coord.shape[0]
    mask = torch.zeros(T, N).cuda()
    stride = 16  # each patch has a shape of 16*16
    img_w = 224
    grids = [(x/(img_w-1), y/(img_w-1)) for x in range(stride//2, img_w, stride) for y in range(stride//2, img_w, stride)]
    assert len(grids) == N-2
    grids = torch.tensor(grids).cuda()  #[196,2]
    coord = coord.unsqueeze(-1).expand(-1,-1,-1,grids.shape[0]).transpose(2,3)  #[T,3,196,2]
    distance = ((coord-grids)**2).sum(dim=-1)  #[T,3,196]
    #We find enough patches to avoid duplicates, top_idx [T,3,3n]
    _, top_idx = distance.topk(k=n_per_region*3, dim=-1, largest=False, sorted=True)
    top_idx = top_idx.transpose(1,2).flatten(start_dim=1) + 2  #[T,6n]
    # print(top_idx.shape)
    for i in range(T):
        idx = top_idx[i].unique(sorted=False)
        idx = idx[idx.shape[0]-3*n_per_region:]  #after unique, the order is inversed.
        mask[i, idx] = 1
        # if (mask[i].sum().item()!=15):
        #     print(idx)

    if patch_rela == 'two_step':
        mask = mask.unsqueeze(-1).bool()  #[T,N,1]
    elif patch_rela == 'one_step':
        mask = mask.unsqueeze(-1)  #[T,N,1]
        complement = (1-mask).transpose(1,2)  #[T,1,N]
        mask = mask.matmul(complement).unsqueeze(1).bool()  #[T,1,N,N]
    else:
        raise ValueError("invalid patch_rela.")
    return mask


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., patch_rela=None, patch_guid=False, n_per_region=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.patch_rela = patch_rela
        self.n_per_region = n_per_region
        self.patch_guid = patch_guid  # 2D mixed Gaussian bias to modulate attention scores of head token
        if patch_guid:
            self.D_layer = nn.Linear(dim, 3)  #sigma=D/2, now we don't learn weights
            self.cen_layer = nn.Linear(dim, 3*2)  #predict center coordinates
            # self.D_layer = nn.Sequential(nn.Linear(dim, dim),
            #                             nn.Tanh(),
            #                             nn.Linear(dim, 3))

    def _gen_mgb(self, coord, D):
        # coord [T,3,2], D [T,3], return mgb [T,1,N,N]
        T = D.shape[0]
        stride = 16
        img_w = 224
        sigma = (D/2).unsqueeze(-1)
        # N = (img_w//stride)**2 + 2
        grids = [(x, y) for x in range(stride//2, img_w, stride) for y in range(stride//2, img_w, stride)]
        assert len(grids) == (img_w//stride)**2
        grids = torch.tensor(grids).cuda()  #[196,2]
        coord = coord * (img_w-1)
        coord = coord.unsqueeze(-1).expand(-1,-1,-1,grids.shape[0]).transpose(2,3)  #[T,3,196,2]
        mgb = ((coord-grids)**2).sum(dim=-1)  #[T,3,196]
        mgb = mgb / (2*sigma**2)
        mgb = torch.exp(-mgb) / 2 / 3.14 / sigma**2
        mgb /= mgb.sum(dim=-1, keepdim=True)
        # mgb = mgb.sum(dim=1)  #[T,196]
        # mgb = mgb.amax(dim=1)
        # mgb = torch.cat((torch.ones(T,2).cuda(), mgb), dim=1)  #[T,N]
        return mgb
    
    def forward(self, x, coord=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.patch_rela == 'one_step':
            mask = gen_ip_mask(self.n_per_region, N, coord, self.patch_rela)  #[B,1,N,N]
            attn = attn.masked_fill(mask, float("-inf"))

        attn = attn.softmax(dim=-1)

        cen = None
        if self.patch_guid:
            img_w = 224
            D = self.D_layer(q.permute(2,0,1,3).reshape(N,B,C)[0])
            D = img_w * torch.sigmoid(D)  #[B,3]
            # D = 56*torch.ones(B,3).cuda()
            cen = self.cen_layer(q.permute(2,0,1,3).reshape(N,B,C)[0])
            cen = torch.sigmoid(cen.reshape(-1,3,2))  #[B,3,2]

            mgb = self._gen_mgb(cen, D)  #[B,3,196]
            mgb = mgb.sum(dim=1) / 3  #same weights
            mgb = torch.cat((torch.ones(B,2).cuda(), mgb), dim=1)  #[B,N]
            # mgb = torch.cat((torch.ones(B,N-1,N).cuda(), mgb.unsqueeze(1)), dim=1)  #[B,N,N]
            # attn = (attn.transpose(1,0)*mgb).transpose(0,1)
            
            mask = torch.zeros(1,1,N,1).bool().cuda()
            mask[0,0,0,0] = True
            head_score = attn.masked_select(mask).reshape(B,self.num_heads,1,N).permute(1,2,0,3)  #[H,1,B,N]
            # print('headsc:', head_score.mean(dim=0)[0,0,2:].amax(), head_score.mean(dim=0)[0,0,2:].amin())
            # head_score = (head_score-mgb).permute(2,0,1,3)  #[B,H,1,N]
            head_score *= mgb
            head_score /= head_score.sum(dim=-1, keepdim=True)
            attn = attn.masked_scatter(mask, head_score)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return cen, attn[:,:,0,:], x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, patch_rela=None, patch_guid=False, n_per_region=5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            patch_rela=patch_rela, patch_guid=patch_guid, n_per_region=n_per_region)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.patch_rela = patch_rela  #another MHA for informative patches only
        if patch_rela == 'two_step':
            self.ip_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, 
                                    patch_rela=patch_rela, patch_guid=patch_guid, n_per_region=n_per_region)
        self.patch_guid = patch_guid  #2D mixed Gaussian to modulate attention scores of the head token
        self.n_per_region = n_per_region  #number of patches per informative region

    def forward(self, x, coord=None):
        T, N, C = x.shape

        if self.patch_rela == 'two_step':
            assert coord is not None  #[T,3,2]
            ip_mask = gen_ip_mask(self.n_per_region, N, coord, self.patch_rela)  #[T,N,1]
            # print(ip_mask.shape)
            ip = x.masked_select(ip_mask).reshape(T, 3*self.n_per_region, C)  #[T,n,c]
            _, ip = self.ip_attn(self.norm1(ip), coord)
            ip = ip + self.drop_path(ip)
            x = x.masked_scatter(ip_mask, ip)

        old_x = x
        cen, attn_score, x = self.attn(self.norm1(x), coord)
        x = old_x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return cen, attn_score, x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        prf = ['one_step', 'one_step', 'one_step', 'one_step', 'one_step', 'one_step', 'one_step', 'one_step']  # patch relation flags
        # prf = [None, None, None, None, None, None, None, None]
        # pgf = [True, True, True, True, True, True, True, True]  # patch gudiance flags
        pgf = [False, False, False, False, False, False, False, False]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                patch_rela=prf[i], patch_guid=pgf[i], n_per_region=16)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, coord=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        attn_score_lst = []
        cen_lst = []
        for blk in self.blocks:
            cen, attn_score, x = blk(x, coord)
            attn_score_lst.append(attn_score)
            cen_lst.append(cen)

        x = self.norm(x)
        return cen_lst, attn_score_lst, x[:, 0]

    def forward(self, x, coord=None):
        attn_score_lst, x = self.forward_features(x, coord)
        x = self.head(x)
        return None, attn_score_lst, x


# def _conv_filter(state_dict, patch_size=16):
#     """ convert patch embedding weight from manual patchify + linear proj to conv"""
#     out_dict = {}
#     for k, v in state_dict.items():
#         if 'patch_embed.proj.weight' in k:
#             v = v.reshape((v.shape[0], 3, patch_size, patch_size))
#         out_dict[k] = v
#     return out_dict


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x, coord=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        attn_score_lst = []
        cen_lst = []
        for blk in self.blocks:
            cen, attn_score, x = blk(x, coord)
            attn_score_lst.append(attn_score)
            cen_lst.append(cen)

        x = self.norm(x)
        return cen_lst, attn_score_lst, x[:, 0], x[:, 1]

    def forward(self, x, **kwargs):
        coord = kwargs.pop('coord', None)
        cen_lst, attn_score_lst, x, x_dist = self.forward_features(x, coord)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        # return None, attn_score_lst, x
        if self.training:
            return {'offset_lst': cen_lst, 'mask_lst': attn_score_lst, 'output': x}
        else:
            # during inference, return the average of both classifier predictions
            return {'offset_lst': cen_lst, 'mask_lst': attn_score_lst, 'output': x}


def deit_small_distilled_patch16_224(pretrained=False, pre_model_path=None, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=8, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(pre_model_path)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model
