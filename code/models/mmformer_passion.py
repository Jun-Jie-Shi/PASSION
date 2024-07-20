import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import fusion_prenorm, general_conv3d_prenorm
from torch.nn.init import constant_, xavier_uniform_
from utils.criterions import temp_kl_loss, softmax_weighted_loss, dice_loss, prototype_passion_loss


basic_dims = 8
### 原论文模型维度
# basic_dims = 16
### 相同基线下模型维度
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_cls = 4
num_modals = 4
patch_size = 5
H = W = Z = 80

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv3d(in_channels=1, out_channels=basic_dims, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=True)
        self.e1_c2 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d_prenorm(basic_dims, basic_dims*2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims*2, pad_type='reflect')
        self.e2_c3 = general_conv3d_prenorm(basic_dims*2, basic_dims*2, pad_type='reflect')

        self.e3_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims*4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*4, pad_type='reflect')
        self.e3_c3 = general_conv3d_prenorm(basic_dims*4, basic_dims*4, pad_type='reflect')

        self.e4_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='reflect')
        self.e4_c3 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='reflect')

        self.e5_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*16, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*16, pad_type='reflect')
        self.e5_c3 = general_conv3d_prenorm(basic_dims*16, basic_dims*16, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        return x1, x2, x3, x4, x5

class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_sep, self).__init__()

        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c1 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.d4_c1(self.d4(x5))

        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred

class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_fuse, self).__init__()

        self.d4_c1 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_d4 = nn.Conv3d(in_channels=basic_dims*16, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims*8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims*4, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims*2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        # self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        # self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        # self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        # self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.RFM5 = fusion_prenorm(in_channel=basic_dims*16, num_cls=num_cls)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims*8, num_cls=num_cls)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims*4, num_cls=num_cls)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims*2, num_cls=num_cls)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims*1, num_cls=num_cls)


    def forward(self, x1, x2, x3, x4, x5, mask=None):
        de_x5_f = self.RFM5(x5)
        # pred4 = self.softmax(self.seg_d4(de_x5_f))
        pred4 = self.seg_d4(de_x5_f)
        de_x5 = self.d4_c1(self.up2(de_x5_f))

        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4_f = self.d4_out(self.d4_c2(de_x4))
        # pred3 = self.softmax(self.seg_d3(de_x4))
        pred3 = self.seg_d3(de_x4_f)
        de_x4 = self.d3_c1(self.up2(de_x4_f))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3_f = self.d3_out(self.d3_c2(de_x3))
        # pred2 = self.softmax(self.seg_d2(de_x3))
        pred2 = self.seg_d2(de_x3_f)
        de_x3 = self.d2_c1(self.up2(de_x3_f))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2_f = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.seg_d1(de_x2_f)
        de_x2 = self.d1_c1(self.up2(de_x2_f))

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1_f = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1_f)
        # pred = self.softmax(logits)
        pred = logits

        return pred, (pred1, pred2, pred3, pred4), (de_x1_f, de_x2_f, de_x3_f, de_x4_f, de_x5_f)


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        # input = x
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(Transformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)


    def forward(self, x, pos):
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x


class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()
    
    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(B, -1, H, W, Z)
        return x


class Model(nn.Module):
    def __init__(self, num_cls=4):
        super(Model, self).__init__()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        ########### IntraFormer
        self.flair_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1ce_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t2_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)

        self.flair_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t1ce_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t1_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t2_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))

        self.flair_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t1ce_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t1_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t2_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        ########### IntraFormer

        ########### InterFormer
        self.multimodal_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim, n_levels=num_modals)
        self.multimodal_decode_conv = nn.Conv3d(transformer_basic_dims*num_modals, basic_dims*16*num_modals, kernel_size=1, padding=0)
        ########### InterFormer

        self.masker = MaskModal()
        self.mask_type = 'idt'
        self.zeros_x1 = torch.zeros(1,basic_dims,H,W,Z).detach()
        self.zeros_x2 = torch.zeros(1,basic_dims*2,H//2,W//2,Z//2).detach()
        self.zeros_x3 = torch.zeros(1,basic_dims*4,H//4,W//4,Z//4).detach()
        self.zeros_x4 = torch.zeros(1,basic_dims*8,H//8,W//8,Z//8).detach()
        self.zeros_x5 = torch.zeros(1,basic_dims*16,H//16,W//16,Z//16).detach()
        self.zeros_intra = torch.zeros(1,transformer_basic_dims,H//16,W//16,Z//16).detach()

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)
        self.up_ops = nn.ModuleList([self.up2, self.up4, self.up8, self.up16])

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

    def forward(self, x, mask, target=None, temp=1.0):
        if self.mask_type == 'pdt':
            #extract feature from different layers
            flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1, :, :, :])
            t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2, :, :, :])
            t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3, :, :, :])
            t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4, :, :, :])

            ########### IntraFormer
            flair_token_x5 = self.flair_encode_conv(flair_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
            t1ce_token_x5 = self.t1ce_encode_conv(t1ce_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
            t1_token_x5 = self.t1_encode_conv(t1_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
            t2_token_x5 = self.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)

            flair_intra_token_x5 = self.flair_transformer(flair_token_x5, self.flair_pos)
            t1ce_intra_token_x5 = self.t1ce_transformer(t1ce_token_x5, self.t1ce_pos)
            t1_intra_token_x5 = self.t1_transformer(t1_token_x5, self.t1_pos)
            t2_intra_token_x5 = self.t2_transformer(t2_token_x5, self.t2_pos)

            flair_intra_x5 = flair_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
            t1ce_intra_x5 = t1ce_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
            t1_intra_x5 = t1_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
            t2_intra_x5 = t2_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        else:
            x = torch.unsqueeze(x, dim=2)
            x = self.masker(x, mask)
            flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1, :, :, :]) if mask[0,0] else (self.zeros_x1.cuda(), self.zeros_x2.cuda(), self.zeros_x3.cuda(), self.zeros_x4.cuda(), self.zeros_x5.cuda())
            t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2, :, :, :]) if mask[0,1] else (self.zeros_x1.cuda(), self.zeros_x2.cuda(), self.zeros_x3.cuda(), self.zeros_x4.cuda(), self.zeros_x5.cuda())
            t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3, :, :, :]) if mask[0,2] else (self.zeros_x1.cuda(), self.zeros_x2.cuda(), self.zeros_x3.cuda(), self.zeros_x4.cuda(), self.zeros_x5.cuda())
            t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4, :, :, :]) if mask[0,3] else (self.zeros_x1.cuda(), self.zeros_x2.cuda(), self.zeros_x3.cuda(), self.zeros_x4.cuda(), self.zeros_x5.cuda())

            if mask[0,0]:
                flair_token_x5 = self.flair_encode_conv(flair_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
                flair_intra_token_x5 = self.flair_transformer(flair_token_x5, self.flair_pos)
                flair_intra_x5 = flair_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
            else:
                flair_intra_x5 = self.zeros_intra.cuda()
            if mask[0,1]:
                t1ce_token_x5 = self.t1ce_encode_conv(t1ce_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
                t1ce_intra_token_x5 = self.t1ce_transformer(t1ce_token_x5, self.t1ce_pos)
                t1ce_intra_x5 = t1ce_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
            else:
                t1ce_intra_x5 = self.zeros_intra.cuda()
            if mask[0,2]:
                t1_token_x5 = self.t1_encode_conv(t1_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
                t1_intra_token_x5 = self.t1_transformer(t1_token_x5, self.t1_pos)
                t1_intra_x5 = t1_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
            else:
                t1_intra_x5 = self.zeros_intra.cuda()
            if mask[0,3]:
                t2_token_x5 = self.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
                t2_intra_token_x5 = self.t2_transformer(t2_token_x5, self.t2_pos)
                t2_intra_x5 = t2_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
            else:
                t2_intra_x5 = self.zeros_intra.cuda()

        if self.is_training:
            if self.mask_type == 'pdt':
                flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
                t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
                t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
                t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
            else:
                flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5) if mask[0,0] else 0
                t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5) if mask[0,1] else 0
                t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5) if mask[0,2] else 0
                t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5) if mask[0,3] else 0
        ########### IntraFormer

        x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask) #Bx4xCxHWZ
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)
        x5_intra = self.masker(torch.stack((flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5), dim=1), mask)

        ########### InterFormer
        flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5 = torch.chunk(x5_intra, num_modals, dim=1)
        multimodal_token_x5 = torch.cat((flair_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                         t1ce_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                         t1_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                         t2_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        ), dim=1)
        multimodal_pos = torch.cat((self.flair_pos, self.t1ce_pos, self.t1_pos, self.t2_pos), dim=1)
        multimodal_inter_token_x5 = self.multimodal_transformer(multimodal_token_x5, multimodal_pos)
        multimodal_inter_x5 = self.multimodal_decode_conv(multimodal_inter_token_x5.view(multimodal_inter_token_x5.size(0), patch_size, patch_size, patch_size, transformer_basic_dims*num_modals).permute(0, 4, 1, 2, 3).contiguous())
        x5_inter = multimodal_inter_x5

        fuse_pred, preds, de_f_avg = self.decoder_fuse(x1, x2, x3, x4, x5_inter)
        ########### InterFormer
        
        if self.is_training:
            if mask[0,0]:
                x1_flair = torch.cat((flair_x1, self.zeros_x1.cuda(), self.zeros_x1.cuda(), self.zeros_x1.cuda()), dim=1) #Bx4xCxHWZ
                x2_flair = torch.cat((flair_x2, self.zeros_x2.cuda(), self.zeros_x2.cuda(), self.zeros_x2.cuda()), dim=1)
                x3_flair = torch.cat((flair_x3, self.zeros_x3.cuda(), self.zeros_x3.cuda(), self.zeros_x3.cuda()), dim=1)
                x4_flair = torch.cat((flair_x4, self.zeros_x4.cuda(), self.zeros_x4.cuda(), self.zeros_x4.cuda()), dim=1)
                flair_token_x5 = torch.cat((flair_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        ), dim=1)
                flair_inter_token_x5 = self.multimodal_transformer(flair_token_x5, multimodal_pos)
                flair_inter_x5 = self.multimodal_decode_conv(flair_inter_token_x5.view(flair_inter_token_x5.size(0), patch_size, patch_size, patch_size, transformer_basic_dims*num_modals).permute(0, 4, 1, 2, 3).contiguous())
                x5_flair = flair_inter_x5
                fuse_pred_flair, preds_flair, de_f_flair = self.decoder_fuse(x1_flair, x2_flair, x3_flair, x4_flair, x5_flair, mask)

            if mask[0,1]:
                x1_t1ce = torch.cat((self.zeros_x1.cuda(), t1ce_x1, self.zeros_x1.cuda(), self.zeros_x1.cuda()), dim=1) #Bx4xCxHWZ
                x2_t1ce = torch.cat((self.zeros_x2.cuda(), t1ce_x2, self.zeros_x2.cuda(), self.zeros_x2.cuda()), dim=1)
                x3_t1ce = torch.cat((self.zeros_x3.cuda(), t1ce_x3, self.zeros_x3.cuda(), self.zeros_x3.cuda()), dim=1)
                x4_t1ce = torch.cat((self.zeros_x4.cuda(), t1ce_x4, self.zeros_x4.cuda(), self.zeros_x4.cuda()), dim=1)
                t1ce_token_x5 = torch.cat((self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        t1ce_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        ), dim=1)
                t1ce_inter_token_x5 = self.multimodal_transformer(t1ce_token_x5, multimodal_pos)
                t1ce_inter_x5 = self.multimodal_decode_conv(t1ce_inter_token_x5.view(t1ce_inter_token_x5.size(0), patch_size, patch_size, patch_size, transformer_basic_dims*num_modals).permute(0, 4, 1, 2, 3).contiguous())
                x5_t1ce = t1ce_inter_x5
                fuse_pred_t1ce, preds_t1ce, de_f_t1ce = self.decoder_fuse(x1_t1ce, x2_t1ce, x3_t1ce, x4_t1ce, x5_t1ce, mask)

            if mask[0,2]:
                x1_t1 = torch.cat((self.zeros_x1.cuda(), self.zeros_x1.cuda(), t1_x1, self.zeros_x1.cuda()), dim=1) #Bx4xCxHWZ
                x2_t1 = torch.cat((self.zeros_x2.cuda(), self.zeros_x2.cuda(), t1_x2, self.zeros_x2.cuda()), dim=1)
                x3_t1 = torch.cat((self.zeros_x3.cuda(), self.zeros_x3.cuda(), t1_x3, self.zeros_x3.cuda()), dim=1)
                x4_t1 = torch.cat((self.zeros_x4.cuda(), self.zeros_x4.cuda(), t1_x4, self.zeros_x4.cuda()), dim=1)
                t1_token_x5 = torch.cat((self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        t1_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        ), dim=1)
                t1_inter_token_x5 = self.multimodal_transformer(t1_token_x5, multimodal_pos)
                t1_inter_x5 = self.multimodal_decode_conv(t1_inter_token_x5.view(t1_inter_token_x5.size(0), patch_size, patch_size, patch_size, transformer_basic_dims*num_modals).permute(0, 4, 1, 2, 3).contiguous())
                x5_t1 = t1_inter_x5
                fuse_pred_t1, preds_t1, de_f_t1 = self.decoder_fuse(x1_t1, x2_t1, x3_t1, x4_t1, x5_t1, mask)

            if mask[0,3]:
                x1_t2 = torch.cat((self.zeros_x1.cuda(), self.zeros_x1.cuda(), self.zeros_x1.cuda(), t2_x1), dim=1) #Bx4xCxHWZ
                x2_t2 = torch.cat((self.zeros_x2.cuda(), self.zeros_x2.cuda(), self.zeros_x2.cuda(), t2_x2), dim=1)
                x3_t2 = torch.cat((self.zeros_x3.cuda(), self.zeros_x3.cuda(), self.zeros_x3.cuda(), t2_x3), dim=1)
                x4_t2 = torch.cat((self.zeros_x4.cuda(), self.zeros_x4.cuda(), self.zeros_x4.cuda(), t2_x4), dim=1)
                t2_token_x5 = torch.cat((self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        t2_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        ), dim=1)
                t2_inter_token_x5 = self.multimodal_transformer(t2_token_x5, multimodal_pos)
                t2_inter_x5 = self.multimodal_decode_conv(t2_inter_token_x5.view(t2_inter_token_x5.size(0), patch_size, patch_size, patch_size, transformer_basic_dims*num_modals).permute(0, 4, 1, 2, 3).contiguous())
                x5_t2 = t2_inter_x5
                fuse_pred_t2, preds_t2, de_f_t2 = self.decoder_fuse(x1_t2, x2_t2, x3_t2, x4_t2, x5_t2, mask)

            kl_loss = torch.zeros(4).cuda().float()
            sep_loss = torch.zeros(4).cuda().float()
            prm_loss = torch.zeros(1).cuda().float()
            proto_loss = torch.zeros(4).cuda().float()
            dist = torch.zeros(4).cuda().float()

            if mask[0,0]:
                sep_loss[0] = softmax_weighted_loss(flair_pred, target, num_cls=num_cls) + dice_loss(flair_pred, target, num_cls=num_cls)
                proto_loss[0], dist[0] = prototype_passion_loss(de_f_flair[0], de_f_avg[0].detach(), target, fuse_pred_flair, fuse_pred.detach(), num_cls=num_cls, temp=temp)
                kl_loss[0] += temp_kl_loss(fuse_pred_flair, fuse_pred.detach(), target, num_cls=num_cls, temp=temp)
                weight_prm = 1.0
                for prm_pred, prm_pred_flair, up_op in zip(preds, preds_flair, self.up_ops):
                    weight_prm /= 2.0
                    kl_loss[0] += weight_prm * temp_kl_loss(prm_pred_flair, prm_pred.detach(), target, num_cls=num_cls, temp=temp, up_op=up_op)
            if mask[0,1]:
                sep_loss[1] = softmax_weighted_loss(t1ce_pred, target, num_cls=num_cls) + dice_loss(t1ce_pred, target, num_cls=num_cls)
                proto_loss[1], dist[1] = prototype_passion_loss(de_f_t1ce[0], de_f_avg[0].detach(), target, fuse_pred_t1ce, fuse_pred.detach(), num_cls=num_cls, temp=temp)
                kl_loss[1] += temp_kl_loss(fuse_pred_t1ce, fuse_pred.detach(), target, num_cls=num_cls, temp=temp)
                weight_prm = 1.0
                for prm_pred, prm_pred_t1ce, up_op in zip(preds, preds_t1ce, self.up_ops):
                    weight_prm /= 2.0
                    kl_loss[1] += weight_prm * temp_kl_loss(prm_pred_t1ce, prm_pred.detach(), target, num_cls=num_cls, temp=temp, up_op=up_op)
            if mask[0,2]:
                sep_loss[2] = softmax_weighted_loss(t1_pred, target, num_cls=num_cls) + dice_loss(t1_pred, target, num_cls=num_cls)
                proto_loss[2], dist[2] = prototype_passion_loss(de_f_t1[0], de_f_avg[0].detach(), target, fuse_pred_t1, fuse_pred.detach(), num_cls=num_cls, temp=temp)
                kl_loss[2] += temp_kl_loss(fuse_pred_t1, fuse_pred.detach(), target, num_cls=num_cls, temp=temp)
                weight_prm = 1.0
                for prm_pred, prm_pred_t1, up_op in zip(preds, preds_t1, self.up_ops):
                    weight_prm /= 2.0
                    kl_loss[2] += weight_prm * temp_kl_loss(prm_pred_t1, prm_pred.detach(), target, num_cls=num_cls, temp=temp, up_op=up_op)
            if mask[0,3]:
                sep_loss[3] = softmax_weighted_loss(t2_pred, target, num_cls=num_cls) + dice_loss(t2_pred, target, num_cls=num_cls)
                proto_loss[3], dist[3] = prototype_passion_loss(de_f_t2[0], de_f_avg[0].detach(), target, fuse_pred_t2, fuse_pred.detach(), num_cls=num_cls, temp=temp)
                kl_loss[3] += temp_kl_loss(fuse_pred_t2, fuse_pred.detach(), target, num_cls=num_cls, temp=temp)
                weight_prm = 1.0
                for prm_pred, prm_pred_t2, up_op in zip(preds, preds_t2, self.up_ops):
                    weight_prm /= 2.0
                    kl_loss[3] += weight_prm * temp_kl_loss(prm_pred_t2, prm_pred.detach(), target, num_cls=num_cls, temp=temp, up_op=up_op)

            weight_prm = 1.0
            for prm_pred, up_op in zip(preds, self.up_ops):
                weight_prm /= 2.0
                prm_loss += weight_prm * softmax_weighted_loss(F.softmax(prm_pred, dim=1), target, num_cls=num_cls, up_op=up_op) \
                + weight_prm * dice_loss(F.softmax(prm_pred, dim=1), target, num_cls=num_cls, up_op=up_op)

            return F.softmax(fuse_pred, dim=1), prm_loss, sep_loss, kl_loss, proto_loss, dist
        return F.softmax(fuse_pred, dim=1)
