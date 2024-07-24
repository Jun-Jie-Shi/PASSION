import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import (general_conv3d, normalization, prm_generator_pk,
                    prm_generator_laststage_pk, region_aware_modal_fusion)
from utils.criterions import temp_kl_loss, softmax_weighted_loss, dice_loss, prototype_passion_loss
from utils.criterions import temp_kl_loss_bs, softmax_weighted_loss_bs, dice_loss_bs, prototype_passion_loss_bs

basic_dims = 8
H = W = Z = 80
num_cls = 4
num_modals = 4
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = general_conv3d(1, basic_dims, pad_type='reflect')
        self.e1_c2 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d(basic_dims, basic_dims*2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d(basic_dims*2, basic_dims*2, pad_type='reflect')
        self.e2_c3 = general_conv3d(basic_dims*2, basic_dims*2, pad_type='reflect')

        self.e3_c1 = general_conv3d(basic_dims*2, basic_dims*4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d(basic_dims*4, basic_dims*4, pad_type='reflect')
        self.e3_c3 = general_conv3d(basic_dims*4, basic_dims*4, pad_type='reflect')

        self.e4_c1 = general_conv3d(basic_dims*4, basic_dims*8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d(basic_dims*8, basic_dims*8, pad_type='reflect')
        self.e4_c3 = general_conv3d(basic_dims*8, basic_dims*8, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        return x1, x2, x3, x4

class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_sep, self).__init__()

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4):
        de_x4 = self.d3_c1(self.d3(x4))

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

        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)

        self.RFM4 = region_aware_modal_fusion(in_channel=basic_dims*8, num_cls=num_cls)
        self.RFM3 = region_aware_modal_fusion(in_channel=basic_dims*4, num_cls=num_cls)
        self.RFM2 = region_aware_modal_fusion(in_channel=basic_dims*2, num_cls=num_cls)
        self.RFM1 = region_aware_modal_fusion(in_channel=basic_dims*1, num_cls=num_cls)
        self.softmax = nn.Softmax(dim=1)

        self.prm_generator4 = prm_generator_laststage_pk(in_channel=basic_dims*8, num_cls=num_cls)
        self.prm_generator3 = prm_generator_pk(in_channel=basic_dims*4, num_cls=num_cls)
        self.prm_generator2 = prm_generator_pk(in_channel=basic_dims*2, num_cls=num_cls)
        self.prm_generator1 = prm_generator_pk(in_channel=basic_dims*1, num_cls=num_cls)


    def forward(self, x1, x2, x3, x4, mask):
        prm_pred4 = self.prm_generator4(x4, mask)
        de_x4 = self.RFM4(x4, self.softmax(prm_pred4).detach(), mask)
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_generator3(de_x4, x3, mask)
        de_x3 = self.RFM3(x3, self.softmax(prm_pred3).detach(), mask)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_generator2(de_x3, x2, mask)
        de_x2 = self.RFM2(x2, self.softmax(prm_pred2).detach(), mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_generator1(de_x2, x1, mask)
        de_x1 = self.RFM1(x1, self.softmax(prm_pred1).detach(), mask)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = logits
        # pred = self.softmax(logits)

        return pred, (prm_pred1, prm_pred2, prm_pred3, prm_pred4), (de_x1, de_x2, de_x3, de_x4)

class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()
    
    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(B, -1, H, W, Z)
        return x

class MaskModal_NoCat(nn.Module):
    def __init__(self):
        super(MaskModal_NoCat, self).__init__()
    
    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y
        return x

class Model(nn.Module):
    def __init__(self, num_cls=4):
        super(Model, self).__init__()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)
        self.masker = MaskModal()
        self.masker_nocat = MaskModal_NoCat()

        self.is_training = False
        self.use_passion = False
        self.mask_type = 'idt'
        self.zeros_x1 = torch.zeros(1,basic_dims,H,W,Z).detach()
        self.zeros_x2 = torch.zeros(1,basic_dims*2,H//2,W//2,Z//2).detach()
        self.zeros_x3 = torch.zeros(1,basic_dims*4,H//4,W//4,Z//4).detach()
        self.zeros_x4 = torch.zeros(1,basic_dims*8,H//8,W//8,Z//8).detach()
        # self.zeros = (self.zeros_x1, self.zeros_x2, self.zeros_x3, self.zeros_x4)
        self.masks_flair = torch.from_numpy(np.array([[True, False, False, False]]))
        self.masks_t1ce = torch.from_numpy(np.array([[False, True, False, False]]))
        self.masks_t1 = torch.from_numpy(np.array([[False, False, True, False]]))
        self.masks_t2 = torch.from_numpy(np.array([[False, False, False, True]]))

        self.masks_mod0 = torch.from_numpy(np.array([[True, False, False, False]])).detach()
        self.masks_mod1 = torch.from_numpy(np.array([[False, True, False, False]])).detach()
        self.masks_mod2 = torch.from_numpy(np.array([[False, False, True, False]])).detach()
        self.masks_mod3 = torch.from_numpy(np.array([[False, False, False, True]])).detach()

        self.up1 = nn.Identity()
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up_ops = nn.ModuleList([self.up1, self.up2, self.up4, self.up8])

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

    def forward(self, x, mask, target=None, temp=1.0):
        B = x.size(0)
        device = x.device
        if self.mask_type == 'pdt':
        # extract feature from different layers
            flair_x1, flair_x2, flair_x3, flair_x4 = self.flair_encoder(x[:, 0:1, :, :, :])
            t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4 = self.t1ce_encoder(x[:, 1:2, :, :, :])
            t1_x1, t1_x2, t1_x3, t1_x4 = self.t1_encoder(x[:, 2:3, :, :, :])
            t2_x1, t2_x2, t2_x3, t2_x4 = self.t2_encoder(x[:, 3:4, :, :, :])

            x1 = torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1) #Bx4xCxHWZ
            x2 = torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1)
            x3 = torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1)
            x4 = torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1)
        else: ### idt or idt_moddrop
            x = torch.unsqueeze(x, dim=2)
            x = self.masker(x, mask)
            flair_x1, flair_x2, flair_x3, flair_x4 = self.flair_encoder(x[:, 0:1, :, :, :])
            t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4 = self.t1ce_encoder(x[:, 1:2, :, :, :])
            t1_x1, t1_x2, t1_x3, t1_x4 = self.t1_encoder(x[:, 2:3, :, :, :])
            t2_x1, t2_x2, t2_x3, t2_x4 = self.t2_encoder(x[:, 3:4, :, :, :])

            x1 = self.masker_nocat(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask) #Bx4xCxHWZ
            x2 = self.masker_nocat(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
            x3 = self.masker_nocat(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
            x4 = self.masker_nocat(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)

        fuse_pred, preds, de_f_avg = self.decoder_fuse(x1, x2, x3, x4, mask)

        if self.is_training:
            if self.mask_type == 'pdt':
                flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4)
                t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4)
                t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4)
                t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4)

            else:
                flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4)
                t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4)
                t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4)
                t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4)

                sep_pred = self.masker(torch.stack((flair_pred, t1ce_pred, t1_pred, t2_pred), dim=1), mask)
                flair_pred, t1ce_pred, t1_pred, t2_pred = torch.chunk(sep_pred, num_modals, dim=1)

            masks_mod0 = torch.tile(self.masks_mod0, [B, 1]).to(device)
            masks_mod1 = torch.tile(self.masks_mod1, [B, 1]).to(device)
            masks_mod2 = torch.tile(self.masks_mod2, [B, 1]).to(device)
            masks_mod3 = torch.tile(self.masks_mod3, [B, 1]).to(device) 

            if self.use_passion:
                ### Mod0-Flair-Path
                fuse_pred_flair, preds_flair, de_f_flair = self.decoder_fuse(x1, x2, x3, x4, masks_mod0)
                ### Mod1-T1c-Path
                fuse_pred_t1ce, preds_t1ce, de_f_t1ce = self.decoder_fuse(x1, x2, x3, x4, masks_mod1)
                ### Mod2-T1-Path
                fuse_pred_t1, preds_t1, de_f_t1 = self.decoder_fuse(x1, x2, x3, x4, masks_mod2)
                ### Mod3-T2-Path
                fuse_pred_t2, preds_t2, de_f_t2 = self.decoder_fuse(x1, x2, x3, x4, masks_mod3)

                ###### Batch-Loss Computation            
                sep_loss = torch.zeros(B,4).float().to(device)
                prm_loss = torch.zeros(B,1).float().to(device)
                kl_loss = torch.zeros(B,4).float().to(device)
                proto_loss = torch.zeros(B,4).float().to(device)
                dist = torch.zeros(B,4).float().to(device)

                weight_prm = 1.0
                for prm_pred, up_op in zip(preds, self.up_ops):
                    weight_prm /= 2.0
                    prm_loss += weight_prm * softmax_weighted_loss_bs(F.softmax(prm_pred, dim=1), target, num_cls=num_cls, up_op=up_op) \
                    + weight_prm * dice_loss_bs(F.softmax(prm_pred, dim=1), target, num_cls=num_cls, up_op=up_op)

                if self.mask_type == 'pdt':
                    ### Mod0-Flair-Path
                    sep_loss += masks_mod0 * (softmax_weighted_loss_bs(flair_pred, target, num_cls=num_cls) + dice_loss_bs(flair_pred, target, num_cls=num_cls))
                    proto_loss_mod0, dist_mod0 = prototype_passion_loss_bs(de_f_flair[0], de_f_avg[0].detach(), target, fuse_pred_flair, fuse_pred.detach(), num_cls=num_cls, temp=temp)
                    proto_loss += masks_mod0 * proto_loss_mod0
                    dist += masks_mod0 * dist_mod0
                    kl_loss += masks_mod0 * temp_kl_loss_bs(fuse_pred_flair, fuse_pred.detach(), target, num_cls=num_cls, temp=temp)
                    weight_prm = 1.0
                    for prm_pred, prm_pred_flair, up_op in zip(preds, preds_flair, self.up_ops):
                        weight_prm /= 2.0
                        kl_loss += masks_mod0 * weight_prm * temp_kl_loss_bs(prm_pred_flair, prm_pred.detach(), target, num_cls=num_cls, temp=temp, up_op=up_op)

                    ### Mod1-T1c-Path
                    sep_loss += masks_mod1 * (softmax_weighted_loss_bs(t1ce_pred, target, num_cls=num_cls) + dice_loss_bs(t1ce_pred, target, num_cls=num_cls))
                    proto_loss_mod1, dist_mod1 = prototype_passion_loss_bs(de_f_t1ce[0], de_f_avg[0].detach(), target, fuse_pred_t1ce, fuse_pred.detach(), num_cls=num_cls, temp=temp)
                    proto_loss += masks_mod1 * proto_loss_mod1
                    dist += masks_mod1 * dist_mod1
                    kl_loss += masks_mod1 * temp_kl_loss_bs(fuse_pred_t1ce, fuse_pred.detach(), target, num_cls=num_cls, temp=temp)
                    weight_prm = 1.0
                    for prm_pred, prm_pred_t1ce, up_op in zip(preds, preds_t1ce, self.up_ops):
                        weight_prm /= 2.0
                        kl_loss += masks_mod1 * weight_prm * temp_kl_loss_bs(prm_pred_t1ce, prm_pred.detach(), target, num_cls=num_cls, temp=temp, up_op=up_op)

                    ### Mod2-T1-Path
                    sep_loss += masks_mod2 * (softmax_weighted_loss_bs(t1_pred, target, num_cls=num_cls) + dice_loss_bs(t1_pred, target, num_cls=num_cls))
                    proto_loss_mod2, dist_mod2 = prototype_passion_loss_bs(de_f_t1[0], de_f_avg[0].detach(), target, fuse_pred_t1, fuse_pred.detach(), num_cls=num_cls, temp=temp)
                    proto_loss += masks_mod2 * proto_loss_mod2
                    dist += masks_mod2 * dist_mod2
                    kl_loss += masks_mod2 * temp_kl_loss_bs(fuse_pred_t1, fuse_pred.detach(), target, num_cls=num_cls, temp=temp)
                    weight_prm = 1.0
                    for prm_pred, prm_pred_t1, up_op in zip(preds, preds_t1, self.up_ops):
                        weight_prm /= 2.0
                        kl_loss += masks_mod2 * weight_prm * temp_kl_loss_bs(prm_pred_t1, prm_pred.detach(), target, num_cls=num_cls, temp=temp, up_op=up_op)

                    ### Mod3-T2-Path
                    sep_loss += masks_mod3 * (softmax_weighted_loss_bs(t2_pred, target, num_cls=num_cls) + dice_loss_bs(t2_pred, target, num_cls=num_cls))
                    proto_loss_mod3, dist_mod3 = prototype_passion_loss_bs(de_f_t2[0], de_f_avg[0].detach(), target, fuse_pred_t2, fuse_pred.detach(), num_cls=num_cls, temp=temp)
                    proto_loss += masks_mod3 * proto_loss_mod3
                    dist += masks_mod3 * dist_mod3
                    kl_loss += masks_mod3 * temp_kl_loss_bs(fuse_pred_t2, fuse_pred.detach(), target, num_cls=num_cls, temp=temp)
                    weight_prm = 1.0
                    for prm_pred, prm_pred_t2, up_op in zip(preds, preds_t2, self.up_ops):
                        weight_prm /= 2.0
                        kl_loss += masks_mod3 * weight_prm * temp_kl_loss_bs(prm_pred_t2, prm_pred.detach(), target, num_cls=num_cls, temp=temp, up_op=up_op)
                else: ### idt or idt_moddrop
                    ### Mod0-Flair-Path
                    sep_loss += mask * masks_mod0 * (softmax_weighted_loss_bs(flair_pred, target, num_cls=num_cls) + dice_loss_bs(flair_pred, target, num_cls=num_cls))
                    proto_loss_mod0, dist_mod0 = prototype_passion_loss_bs(de_f_flair[0], de_f_avg[0].detach(), target, fuse_pred_flair, fuse_pred.detach(), num_cls=num_cls, temp=temp)
                    proto_loss += mask * masks_mod0 * proto_loss_mod0
                    dist += mask * masks_mod0 * dist_mod0
                    kl_loss += mask * masks_mod0 * temp_kl_loss_bs(fuse_pred_flair, fuse_pred.detach(), target, num_cls=num_cls, temp=temp)
                    weight_prm = 1.0
                    for prm_pred, prm_pred_flair, up_op in zip(preds, preds_flair, self.up_ops):
                        weight_prm /= 2.0
                        kl_loss += mask * masks_mod0 * weight_prm * temp_kl_loss_bs(prm_pred_flair, prm_pred.detach(), target, num_cls=num_cls, temp=temp, up_op=up_op)

                    ### Mod1-T1c-Path
                    sep_loss += mask * masks_mod1 * (softmax_weighted_loss_bs(t1ce_pred, target, num_cls=num_cls) + dice_loss_bs(t1ce_pred, target, num_cls=num_cls))
                    proto_loss_mod1, dist_mod1 = prototype_passion_loss_bs(de_f_t1ce[0], de_f_avg[0].detach(), target, fuse_pred_t1ce, fuse_pred.detach(), num_cls=num_cls, temp=temp)
                    proto_loss += mask * masks_mod1 * proto_loss_mod1
                    dist += mask * masks_mod1 * dist_mod1
                    kl_loss += mask * masks_mod1 * temp_kl_loss_bs(fuse_pred_t1ce, fuse_pred.detach(), target, num_cls=num_cls, temp=temp)
                    weight_prm = 1.0
                    for prm_pred, prm_pred_t1ce, up_op in zip(preds, preds_t1ce, self.up_ops):
                        weight_prm /= 2.0
                        kl_loss += mask * masks_mod1 * weight_prm * temp_kl_loss_bs(prm_pred_t1ce, prm_pred.detach(), target, num_cls=num_cls, temp=temp, up_op=up_op)

                    ### Mod2-T1-Path
                    sep_loss += mask * masks_mod2 * (softmax_weighted_loss_bs(t1_pred, target, num_cls=num_cls) + dice_loss_bs(t1_pred, target, num_cls=num_cls))
                    proto_loss_mod2, dist_mod2 = prototype_passion_loss_bs(de_f_t1[0], de_f_avg[0].detach(), target, fuse_pred_t1, fuse_pred.detach(), num_cls=num_cls, temp=temp)
                    proto_loss += mask * masks_mod2 * proto_loss_mod2
                    dist += mask * masks_mod2 * dist_mod2
                    kl_loss += mask * masks_mod2 * temp_kl_loss_bs(fuse_pred_t1, fuse_pred.detach(), target, num_cls=num_cls, temp=temp)
                    weight_prm = 1.0
                    for prm_pred, prm_pred_t1, up_op in zip(preds, preds_t1, self.up_ops):
                        weight_prm /= 2.0
                        kl_loss += mask * masks_mod2 * weight_prm * temp_kl_loss_bs(prm_pred_t1, prm_pred.detach(), target, num_cls=num_cls, temp=temp, up_op=up_op)

                    ### Mod3-T2-Path
                    sep_loss += mask * masks_mod3 * (softmax_weighted_loss_bs(t2_pred, target, num_cls=num_cls) + dice_loss_bs(t2_pred, target, num_cls=num_cls))
                    proto_loss_mod3, dist_mod3 = prototype_passion_loss_bs(de_f_t2[0], de_f_avg[0].detach(), target, fuse_pred_t2, fuse_pred.detach(), num_cls=num_cls, temp=temp)
                    proto_loss += mask * masks_mod3 * proto_loss_mod3
                    dist += mask * masks_mod3 * dist_mod3
                    kl_loss += mask * masks_mod3 * temp_kl_loss_bs(fuse_pred_t2, fuse_pred.detach(), target, num_cls=num_cls, temp=temp)
                    weight_prm = 1.0
                    for prm_pred, prm_pred_t2, up_op in zip(preds, preds_t2, self.up_ops):
                        weight_prm /= 2.0
                        kl_loss += mask * masks_mod3 * weight_prm * temp_kl_loss_bs(prm_pred_t2, prm_pred.detach(), target, num_cls=num_cls, temp=temp, up_op=up_op)

                return F.softmax(fuse_pred, dim=1), prm_loss, sep_loss, kl_loss, proto_loss, dist

            else: ### without passion
                sep_loss = torch.zeros(B,4).float().to(device)
                prm_loss = torch.zeros(B,1).float().to(device)

                weight_prm = 1.0
                for prm_pred, up_op in zip(preds, self.up_ops):
                    weight_prm /= 2.0
                    prm_loss += weight_prm * softmax_weighted_loss_bs(F.softmax(prm_pred, dim=1), target, num_cls=num_cls, up_op=up_op) \
                    + weight_prm * dice_loss_bs(F.softmax(prm_pred, dim=1), target, num_cls=num_cls, up_op=up_op)

                if self.mask_type == 'pdt':
                    sep_loss += masks_mod0 * (softmax_weighted_loss_bs(flair_pred, target, num_cls=num_cls) + dice_loss_bs(flair_pred, target, num_cls=num_cls))
                    sep_loss += masks_mod1 * (softmax_weighted_loss_bs(t1ce_pred, target, num_cls=num_cls) + dice_loss_bs(t1ce_pred, target, num_cls=num_cls))
                    sep_loss += masks_mod2 * (softmax_weighted_loss_bs(t1_pred, target, num_cls=num_cls) + dice_loss_bs(t1_pred, target, num_cls=num_cls))
                    sep_loss += masks_mod3 * (softmax_weighted_loss_bs(t2_pred, target, num_cls=num_cls) + dice_loss_bs(t2_pred, target, num_cls=num_cls))
                else: ### idt or idt_moddrop
                    sep_loss += mask * masks_mod0 * (softmax_weighted_loss_bs(flair_pred, target, num_cls=num_cls) + dice_loss_bs(flair_pred, target, num_cls=num_cls))
                    sep_loss += mask * masks_mod1 * (softmax_weighted_loss_bs(t1ce_pred, target, num_cls=num_cls) + dice_loss_bs(t1ce_pred, target, num_cls=num_cls))
                    sep_loss += mask * masks_mod2 * (softmax_weighted_loss_bs(t1_pred, target, num_cls=num_cls) + dice_loss_bs(t1_pred, target, num_cls=num_cls))
                    sep_loss += mask * masks_mod3 * (softmax_weighted_loss_bs(t2_pred, target, num_cls=num_cls) + dice_loss_bs(t2_pred, target, num_cls=num_cls))

                return F.softmax(fuse_pred, dim=1), prm_loss, sep_loss
        return F.softmax(fuse_pred, dim=1)
                

