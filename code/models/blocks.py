import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.mask import mask_gen_cross4

basic_dims = 16
num_modals = 4

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    # elif norm == 'sync_bn':
    #     m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

def nchwd2nlc2nchwd(module, x):
    B, C, H, W, D = x.shape
    x = x.flatten(2).transpose(1, 2)
    x = module(x)
    x = x.transpose(1, 2).reshape(B, C, H, W, D).contiguous()
    return x

class DepthWiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthWiseConvBlock, self).__init__()
        mid_channels = in_channels
        self.conv1 = nn.Conv3d(in_channels,
                               mid_channels,
                               1, 1)
        layer_norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = layer_norm(mid_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv3d(mid_channels,
                               mid_channels,
                               3, 1, 1, groups=mid_channels)
        self.norm2 = layer_norm(mid_channels)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv3d(mid_channels,
                               out_channels,
                               1, 1)
        self.norm3 = layer_norm(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = nchwd2nlc2nchwd(self.norm1, x)
        x = self.act1(x)

        x = self.conv2(x)
        x = nchwd2nlc2nchwd(self.norm2, x)
        x = self.act2(x)

        x = self.conv3(x)
        x = nchwd2nlc2nchwd(self.norm3, x)
        return x

class GroupConvBlock(nn.Module):
    def __init__(self,
                 embed_dims=basic_dims,
                 expand_ratio=4,
                 proj_drop=0.):
        super(GroupConvBlock, self).__init__()
        self.pwconv1 = nn.Conv3d(embed_dims,
                                 embed_dims * expand_ratio,
                                 1, 1)
        layer_norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = layer_norm(embed_dims * expand_ratio)
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv3d(embed_dims * expand_ratio,
                                embed_dims * expand_ratio,
                                3, 1, 1, groups=embed_dims)
        self.norm2 = layer_norm(embed_dims * expand_ratio)
        self.act2 = nn.GELU()
        self.pwconv2 = nn.Conv3d(embed_dims * expand_ratio,
                                 embed_dims,
                                 1, 1)
        self.norm3 = layer_norm(embed_dims)
        self.final_act = nn.GELU()
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, identity=None):
        input = x
        x = self.pwconv1(x)
        x = nchwd2nlc2nchwd(self.norm1, x)
        x = self.act1(x)

        x = self.dwconv(x)
        x = nchwd2nlc2nchwd(self.norm2, x)
        x = self.act2(x)

        x = self.pwconv2(x)
        x = nchwd2nlc2nchwd(self.norm3, x)

        if identity is None:
            x = input + self.proj_drop(x)
        else:
            x = identity + self.proj_drop(x)

        x = self.final_act(x)

        return x

class AttentionLayer(nn.Module):
    def __init__(self,
                 kv_dim=basic_dims,
                 query_dim=num_modals,
                 attn_drop=0.,
                 proj_drop=0.):
        super(AttentionLayer, self).__init__()
        self.attn_drop = nn.Dropout(attn_drop)
        self.query_map = DepthWiseConvBlock(query_dim, query_dim)
        self.key_map = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map = DepthWiseConvBlock(kv_dim, kv_dim)
        self.out_project = DepthWiseConvBlock(query_dim, query_dim)

        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, query, key, value):
        """x: B, C, H, W, D"""
        identity = query
        qb, qc, qh, qw, qd = query.shape
        query = self.query_map(query).flatten(2)
        key = self.key_map(key).flatten(2)
        value = self.value_map(value).flatten(2)

        attn = (query @ key.transpose(-2, -1)) * (query.shape[-1]) ** -0.5
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ value
        x = x.reshape(qb, qc, qh, qw, qd)
        x = self.out_project(x)
        return identity + self.proj_drop(x)


class CrossBlock(nn.Module):
    def __init__(self,
                 feature_channels=basic_dims,
                 num_classes=num_modals,
                 expand_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 ffn_feature_maps=True):
        super(CrossBlock, self).__init__()
        self.ffn_feature_maps = ffn_feature_maps

        self.cross_attn = AttentionLayer(kv_dim=feature_channels,
                                         query_dim=num_classes,
                                         attn_drop=attn_drop_rate,
                                         proj_drop=drop_rate)

        if ffn_feature_maps:
            self.ffn2 = GroupConvBlock(embed_dims=feature_channels,
                                       expand_ratio=expand_ratio)
        self.ffn1 = GroupConvBlock(embed_dims=num_classes,
                                   expand_ratio=expand_ratio)

    def forward(self, kernels, feature_maps):
        kernels = self.cross_attn(query=kernels,
                                  key=feature_maps,
                                  value=feature_maps)

        kernels = self.ffn1(kernels, identity=kernels)

        if self.ffn_feature_maps:
            feature_maps = self.ffn2(feature_maps, identity=feature_maps)

        return kernels, feature_maps

class ResBlock(nn.Module):
    def __init__(self, in_channels=4, channels=4):
        super(ResBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_channels, eps=1e-6)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv3d(in_channels, channels, 3, 1, 1)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv3d(channels, channels, 3, 1, 1)
        if channels != in_channels:
            self.identity_map = nn.Conv3d(in_channels, channels, 1, 1, 0)
        else:
            self.identity_map = nn.Identity()

    def forward(self, x):
        # refer to paper
        # Identity Mapping in Deep Residual Networks
        out = nchwd2nlc2nchwd(self.norm1, x)
        out = self.act1(out)
        out = self.conv1(out)
        out = nchwd2nlc2nchwd(self.norm2, out)
        out = self.act2(out)
        out = self.conv2(out)
        out = out + self.identity_map(x)

        return out

class MultiMaskCrossBlock(nn.Module):
    def __init__(self,
                 feature_channels=basic_dims*16,
                 num_classes=basic_dims*16,
                 expand_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 ffn_feature_maps=True):
        super(MultiMaskCrossBlock, self).__init__()
        self.ffn_feature_maps = ffn_feature_maps

        self.cross_attn = MultiMaskAttentionLayer(kv_dim=feature_channels,
                                         query_dim=num_classes,
                                         attn_drop=attn_drop_rate,
                                         proj_drop=drop_rate)

        if ffn_feature_maps:
            self.ffn2 = GroupConvBlock(embed_dims=feature_channels,
                                       expand_ratio=expand_ratio)
        self.ffn1 = GroupConvBlock(embed_dims=num_classes,
                                   expand_ratio=expand_ratio)

    def forward(self, kernels, feature_maps, mask):
        flair, t1ce, t1, t2 = feature_maps
        kernels = self.cross_attn(query = kernels,
                                  key = feature_maps,
                                  value = feature_maps,
                                  mask = mask)

        kernels = self.ffn1(kernels, identity=kernels)

        if self.ffn_feature_maps:
            flair = self.ffn2(flair, identity=flair)
            t1ce = self.ffn2(t1ce, identity=t1ce)
            t1 = self.ffn2(t1, identity=t1)
            t2 = self.ffn2(t2, identity=t2)
            feature_maps = (flair, t1ce, t1, t2)

        return kernels, feature_maps

class MultiMaskAttentionLayer(nn.Module):
    def __init__(self,
                 kv_dim=basic_dims,
                 query_dim=num_modals,
                 attn_drop=0.,
                 proj_drop=0.):
        super(MultiMaskAttentionLayer, self).__init__()
        self.attn_drop = nn.Dropout(attn_drop)
        self.query_map = DepthWiseConvBlock(query_dim, query_dim)
        self.key_map_flair = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_flair = DepthWiseConvBlock(kv_dim, kv_dim)
        self.key_map_t1ce = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_t1ce = DepthWiseConvBlock(kv_dim, kv_dim)
        self.key_map_t1 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_t1 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.key_map_t2 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_t2 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.out_project = DepthWiseConvBlock(query_dim, query_dim)

        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, query, key, value, mask):
        """x: B, C, H, W, D"""
        identity = query
        flair, t1ce, t1, t2 = key
        qb, qc, qh, qw, qd = query.shape
        query = self.query_map(query).flatten(2)
        key_flair = self.key_map_flair(flair).flatten(2)
        value_flair = self.value_map_flair(flair).flatten(2)
        key_t1ce = self.key_map_t1ce(t1ce).flatten(2)
        value_t1ce = self.value_map_t1ce(t1ce).flatten(2)
        key_t1 = self.key_map_t1(t1).flatten(2)
        value_t1 = self.value_map_t1(t1).flatten(2)
        key_t2 = self.key_map_t2(t2).flatten(2)
        value_t2 = self.value_map_t2(t2).flatten(2)

        key = torch.cat((key_flair, key_t1ce, key_t1, key_t2), dim=1)
        value = torch.cat((value_flair, value_t1ce, value_t1, value_t2), dim=1)

        kb, kc, kl = key.shape

        attn = (query @ key.transpose(-2, -1)) * (query.shape[-1]) ** -0.5
        self_mask = mask_gen_cross4(qb, qc, kc, mask).cuda(non_blocking=True)
        attn = attn.masked_fill(self_mask==0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ value
        x = x.reshape(qb, qc, qh, qw, qd)
        x = self.out_project(x)
        return identity + self.proj_drop(x)


class general_conv3d_prenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv3d_prenorm, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)


    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x

class general_prenorm_noconv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_prenorm_noconv3d, self).__init__()
        # self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)


    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        # x = self.conv(x)
        return x

class general_conv3d_noprenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv3d_noprenorm, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        # self.norm = normalization(out_ch, norm=norm)
        # if act_type == 'relu':
        #     self.activation = nn.ReLU(inplace=True)
        # elif act_type == 'lrelu':
        #     self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)


    def forward(self, x):
        # x = self.norm(x)
        # x = self.activation(x)
        x = self.conv(x)
        return x

class general_conv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='reflect', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class prm_generator_laststage(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):
        super(prm_generator_laststage, self).__init__()

        self.embedding_layer = nn.Sequential(
                            general_conv3d(in_channel*4, int(in_channel//4), k_size=1, padding=0, stride=1),
                            general_conv3d(int(in_channel//4), int(in_channel//4), k_size=3, padding=1, stride=1),
                            general_conv3d(int(in_channel//4), in_channel, k_size=1, padding=0, stride=1))

        self.prm_layer = nn.Sequential(
                            general_conv3d(in_channel, 16, k_size=1, stride=1, padding=0),
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
                            nn.Softmax(dim=1))

    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        y = y.view(B, -1, H, W, Z)

        seg = self.prm_layer(self.embedding_layer(y))
        return seg

### prm_generator_laststage without softmax
class prm_generator_laststage_pk(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):
        super(prm_generator_laststage_pk, self).__init__()

        self.embedding_layer = nn.Sequential(
                            general_conv3d(in_channel*4, int(in_channel//4), k_size=1, padding=0, stride=1),
                            general_conv3d(int(in_channel//4), int(in_channel//4), k_size=3, padding=1, stride=1),
                            general_conv3d(int(in_channel//4), in_channel, k_size=1, padding=0, stride=1))

        self.prm_layer = nn.Sequential(
                            general_conv3d(in_channel, 16, k_size=1, stride=1, padding=0),
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True))

    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        y = y.view(B, -1, H, W, Z)

        logit = self.prm_layer(self.embedding_layer(y))
        return logit

class prm_generator(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):
        super(prm_generator, self).__init__()

        self.embedding_layer = nn.Sequential(
                            general_conv3d(in_channel*4, int(in_channel//4), k_size=1, padding=0, stride=1),
                            general_conv3d(int(in_channel//4), int(in_channel//4), k_size=3, padding=1, stride=1),
                            general_conv3d(int(in_channel//4), in_channel, k_size=1, padding=0, stride=1))


        self.prm_layer = nn.Sequential(
                            general_conv3d(in_channel*2, 16, k_size=1, stride=1, padding=0),
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
                            nn.Softmax(dim=1))

    def forward(self, x1, x2, mask):
        B, K, C, H, W, Z = x2.size()
        y = torch.zeros_like(x2)
        y[mask, ...] = x2[mask, ...]
        y = y.view(B, -1, H, W, Z)

        seg = self.prm_layer(torch.cat((x1, self.embedding_layer(y)), dim=1))
        return seg

### prm_generator without softmax
class prm_generator_pk(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):
        super(prm_generator_pk, self).__init__()

        self.embedding_layer = nn.Sequential(
                            general_conv3d(in_channel*4, int(in_channel//4), k_size=1, padding=0, stride=1),
                            general_conv3d(int(in_channel//4), int(in_channel//4), k_size=3, padding=1, stride=1),
                            general_conv3d(int(in_channel//4), in_channel, k_size=1, padding=0, stride=1))


        self.prm_layer = nn.Sequential(
                            general_conv3d(in_channel*2, 16, k_size=1, stride=1, padding=0),
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True))

    def forward(self, x1, x2, mask):
        B, K, C, H, W, Z = x2.size()
        y = torch.zeros_like(x2)
        y[mask, ...] = x2[mask, ...]
        y = y.view(B, -1, H, W, Z)

        logit = self.prm_layer(torch.cat((x1, self.embedding_layer(y)), dim=1))
        return logit

class prm_fusion(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):
        super(prm_fusion, self).__init__()

        self.prm_layer = nn.Sequential(
                            general_conv3d(in_channel, 16, k_size=1, stride=1, padding=0),
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
                            nn.Softmax(dim=1))

    def forward(self, x1):

        seg = self.prm_layer(x1)
        return seg

### prm_fusion without softmax
class prm_fusion_pk(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):
        super(prm_fusion_pk, self).__init__()

        self.prm_layer = nn.Sequential(
                            general_conv3d(in_channel, 16, k_size=1, stride=1, padding=0),
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True))

    def forward(self, x1):

        logit = self.prm_layer(x1)
        return logit

####modal fusion in each region
class modal_fusion(nn.Module):
    def __init__(self, in_channel=64):
        super(modal_fusion, self).__init__()
        self.weight_layer = nn.Sequential(
                            nn.Conv3d(4*in_channel+1, 128, 1, padding=0, bias=True),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv3d(128, 4, 1, padding=0, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prm, region_name):
        B, K, C, H, W, Z = x.size()

        prm_avg = torch.mean(prm, dim=(3,4,5), keepdim=False) + 1e-7
        feat_avg = torch.mean(x, dim=(3,4,5), keepdim=False) / prm_avg

        feat_avg = feat_avg.view(B, K*C, 1, 1, 1)
        feat_avg = torch.cat((feat_avg, prm_avg[:, 0, 0, ...].view(B, 1, 1, 1, 1)), dim=1)
        weight = torch.reshape(self.weight_layer(feat_avg), (B, K, 1))
        weight = self.sigmoid(weight).view(B, K, 1, 1, 1, 1)

        ###we find directly using weighted sum still achieve competing performance
        region_feat = torch.sum(x * weight, dim=1)
        return region_feat

###fuse region feature
class region_fusion(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(region_fusion, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d(in_channel, in_channel//2, k_size=1, padding=0, stride=1))

    def forward(self, x):
        B, _, _, H, W, Z = x.size()
        x = torch.reshape(x, (B, -1, H, W, Z))
        return self.fusion_layer(x)

class fusion_prenorm(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(fusion_prenorm, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d_prenorm(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x):
        return self.fusion_layer(x)
    
class fusion_conv3d_prenorm(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(fusion_conv3d_prenorm, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d_noprenorm(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x):
        return self.fusion_layer(x)

class fusion_conv3d_prenormplus(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(fusion_conv3d_prenormplus, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d_noprenorm(in_channel*num_cls, in_channel*num_cls, k_size=1, padding=0, stride=1),
                        general_conv3d_prenorm(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x):
        return self.fusion_layer(x)

class fusion_postnorm(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(fusion_postnorm, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        y = y.view(B, -1, H, W, Z)
        return self.fusion_layer(y)

class region_aware_modal_fusion(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):
        super(region_aware_modal_fusion, self).__init__()
        self.num_cls = num_cls

        self.modal_fusion = nn.ModuleList([modal_fusion(in_channel=in_channel) for i in range(num_cls)])
        self.region_fusion = region_fusion(in_channel=in_channel, num_cls=num_cls)
        self.short_cut = nn.Sequential(
                        general_conv3d(in_channel*4, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d(in_channel, in_channel//2, k_size=1, padding=0, stride=1))

        self.clsname_list = ['BG', 'NCR/NET', 'ED', 'ET'] ##BRATS2020 and BRATS2018
        self.clsname_list = ['BG', 'NCR', 'ED', 'NET', 'ET'] ##BRATS2015

    def forward(self, x, prm, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]

        prm = torch.unsqueeze(prm, 2).repeat(1, 1, C, 1, 1, 1)
        ###divide modal features into different regions
        flair = y[:, 0:1, ...] * prm
        t1ce = y[:, 1:2, ...] * prm
        t1 = y[:, 2:3, ...] * prm
        t2 = y[:, 3:4, ...] * prm

        modal_feat = torch.stack((flair, t1ce, t1, t2), dim=1)
        region_feat = [modal_feat[:, :, i, :, :] for i in range(self.num_cls)]

        ###modal fusion in each region
        region_fused_feat = []
        for i in range(self.num_cls):
            region_fused_feat.append(self.modal_fusion[i](region_feat[i], prm[:, i:i+1, ...], self.clsname_list[i]))
        region_fused_feat = torch.stack(region_fused_feat, dim=1)
        '''
        region_fused_feat = torch.stack((self.modal_fusion[0](region_feat[0], prm[:, 0:1, ...], 'BG'),
                                         self.modal_fusion[1](region_feat[1], prm[:, 1:2, ...], 'NCR/NET'),
                                         self.modal_fusion[2](region_feat[2], prm[:, 2:3, ...], 'ED'),
                                         self.modal_fusion[3](region_feat[3], prm[:, 3:4, ...], 'ET')), dim=1)
        '''

        ###gain final feat with a short cut
        final_feat = torch.cat((self.region_fusion(region_fused_feat), self.short_cut(y.view(B, -1, H, W, Z))), dim=1)
        return final_feat
