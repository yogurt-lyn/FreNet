import math
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from functools import partial
from importlib import import_module
import matplotlib.pyplot as plt

from timm.layers import DropPath, to_2tuple, trunc_normal_
from basemodel.build_sam import sam_model_registry


class ImplicitPriorNet(nn.Module):
    def __init__(self, in_channels, latent_dim=128):
        super().__init__()
        # 坐标编码器
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim))
        
        # 隐式解码器
        self.implicit_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, in_channels))
        
        # 先验条件生成器
        self.prior_conditioner = nn.Sequential(
            nn.Conv2d(1, latent_dim, 3, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x, seg_prior):
        B, C, H, W = x.shape
        # 生成网格坐标
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
        coords = torch.stack([grid_x, grid_y], dim=-1).to(x.device).repeat(B, 1, 1, 1)
        
        # 编码坐标
        coord_feat = self.coord_encoder(coords.view(B*H*W, 2))
        
        # 从分割先验生成条件向量
        condition = self.prior_conditioner(seg_prior).view(B, 1, -1)
        condition = condition.repeat(1, H*W, 1).view(B*H*W, -1)
        
        # 融合坐标特征和条件
        fused_feat = torch.cat([coord_feat, condition], dim=1)

        # 解码为特征调制参数
        mod_params = self.implicit_decoder(fused_feat).view(B, H, W, C).permute(0, 3, 1, 2)
        
        # 应用调制
        return x * mod_params.sigmoid() + mod_params
    

# Spacial Enhence  
class SpacialLoclization(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super().__init__()

    def forward(self, pvt_feat, mask_feat):
        B, C, H, W = pvt_feat.shape

        # 逐点余弦相似度计算
        pvt_flat = F.normalize(pvt_feat.view(B, C, -1), dim=1)      # (B, C, H*W)
        mask_flat = F.normalize(mask_feat.view(B, C, -1), dim=1)    # (B, C, H*W)

        # 计算余弦相似度 (B, 1, H, W)
        cos_sim = (pvt_flat * mask_flat).sum(dim=1).view(B, 1, H, W)
        attention = torch.sigmoid(cos_sim)  # 位置先验

        # 融合
        enhanced_feat = attention * pvt_feat
        resby=nn.ReLU(inplace=True)(pvt_feat-enhanced_feat)
        out=enhanced_feat+resby
        return out


# Frequence Enhence
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        
        # 空间注意力
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_attention = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attention

        avg_out = torch.mean(x, dim=1, keepdim=True)        
        max_out, _ = torch.max(x, dim=1, keepdim=True)         
        spatial = torch.cat([avg_out, max_out], dim=1)         
        spatial_attention = self.sigmoid_spatial(self.conv_spatial(spatial))  
        x = x * spatial_attention                           
        return x


def freq_magnitude_map(d1, d2, use_rfft=False, device='cpu'):
    freq_h = torch.fft.fftfreq(d1, device=device)
    freq_w = torch.fft.rfftfreq(d2, device=device) if use_rfft else torch.fft.fftfreq(d2, device=device)
    mesh_h, mesh_w = torch.meshgrid(freq_h, freq_w, indexing='ij')
    freq_hw = torch.stack([mesh_h, mesh_w], dim=-1)  # shape: (H, W, 2)
    freq_magnitude = torch.norm(freq_hw, dim=-1)     # shape: (H, W)
    return freq_magnitude, freq_hw

class FrequenceBandModulation(nn.Module):
    def __init__(self, in_channels, band_scales=[2, 4, 8],conv_kernel=3, groups=1, weight_init='zero'):
        super().__init__()
        self.band_scales = band_scales
        self.groups = groups

        self.mask_fuse_alpha = nn.Parameter(torch.tensor(0.5))  # 结构 vs 边缘
        self.mask_fuse_beta = nn.Parameter(torch.tensor(0.5))   # 原始mask vs 更新mask


        self.in_channels = in_channels * 2  # 适应concat后的通道数
        self.num_bands = len(band_scales) + 1

        self.band_convs = nn.ModuleList()
        for _ in range(self.num_bands):
            conv = nn.Conv2d(self.in_channels, in_channels, kernel_size=conv_kernel,
                             padding=conv_kernel // 2, groups=groups)
            if weight_init == 'zero':
                nn.init.normal_(conv.weight, std=1e-6)
                nn.init.constant_(conv.bias, 0)
            self.band_convs.append(conv)
        self.band_weights = nn.Parameter(torch.ones(self.num_bands), requires_grad=True)
        self.attention_blocks = nn.ModuleList([
            CBAM(in_channels) for _ in range(self.num_bands)
        ])

        self.fusion_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Conv2d(in_channels, self.num_bands, kernel_size=1),  
            nn.Softmax(dim=1)
        )
        self.activate = lambda x: 2 * torch.sigmoid(x)
  
    def forward(self, x, mask=None, context=None):
        b, c, h, w = x.shape

        context = torch.cat([x, mask], dim=1)

        # 傅里叶变换
        x_fft = torch.fft.rfft2(x, norm='ortho')
        x_band = x.clone()
        outputs = []
        B, C, H, W = x_fft.shape  # rfft2 后 W 是原来宽度的一半+1

        # 获取频率幅度图
        freq_magnitude, _ = freq_magnitude_map(h, w, use_rfft=True, device=x.device)
        freq_max = freq_magnitude.max()

        # 构建频率区间
        thresholds = [freq_max / s for s in self.band_scales]
        thresholds = [0.0] + thresholds  
        thresholds = sorted(thresholds)  

        for idx in range(self.num_bands):
            low = thresholds[idx]
            high = thresholds[idx + 1] if idx + 1 < len(thresholds) else freq_max + 1e-6

            band_mask = ((freq_magnitude > low) & (freq_magnitude <= high)).float()
            
            freq_mask = band_mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
            low_part = torch.fft.irfft2(x_fft * freq_mask, s=(h, w), norm='ortho').real
            mask_part=torch.fft.irfft2(freq_mask, s=(h, w), norm='ortho').real
            high_part = x_band - low_part

            weight_map = self.activate(self.band_convs[idx](context))
            modulated = weight_map * high_part
            x_fft = torch.fft.rfft2(modulated, norm='ortho') 
            low_part = self.attention_blocks[idx](low_part)
            outputs.append(low_part)

        norm_weights = torch.softmax(self.band_weights, dim=0)  # shape: [num_bands]
        output = 0
        for i in range(self.num_bands):
            output = output + norm_weights[i] * outputs[i]
        
         # ====== Mask更新部分 ======
        if mask is not None:
            # 高频边缘增强
            edge_detail = outputs[-1]
            edge_score = torch.sigmoid(edge_detail.mean(dim=1, keepdim=True))

            # 低频结构补全（使用 CBMA 后的低频特征）
            structure_feat = outputs[0]
            structure_score = torch.sigmoid(structure_feat.mean(dim=1, keepdim=True))

            # 融合两个来源（可学习的权重）
            if hasattr(self, 'mask_fuse_alpha') and hasattr(self, 'mask_fuse_beta'):
                mask_update = self.mask_fuse_alpha * structure_score + (1 - self.mask_fuse_alpha) * edge_score
                updated_mask = self.mask_fuse_beta * mask + (1 - self.mask_fuse_beta) * mask_update
            else:
                # 默认融合比例
                mask_update = 0.5 * structure_score + 0.5 * edge_score
                updated_mask = 0.5 * mask + 0.5 * mask_update

            return output, updated_mask


class FPNFusion(nn.Module):
    def __init__(self, pvt_channels):
        super().__init__()
        self.pvt_channels = pvt_channels  

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(pvt_channels[1], pvt_channels[0], kernel_size=1),
            nn.Conv2d(pvt_channels[2], pvt_channels[1], kernel_size=1),
            nn.Conv2d(pvt_channels[3], pvt_channels[2], kernel_size=1)
        ])
       
        self.conv1 = nn.Sequential(
            nn.Conv2d(pvt_channels[1], pvt_channels[0], kernel_size=1),
            nn.BatchNorm2d(pvt_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(pvt_channels[2], pvt_channels[1], kernel_size=1),
            nn.BatchNorm2d(pvt_channels[1]),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(pvt_channels[3], pvt_channels[2], kernel_size=1),
            nn.BatchNorm2d(pvt_channels[2]),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):  # features = [p1, p2, p3, p4]
        p1, p2, p3, p4 = features  # p1: lowest level, p4: highest level

        # top-down fusion
        p4 = self.conv3(p4)
        p3 = p3 + F.interpolate(p4, size=p3.shape[2:], mode="nearest")
        p2 = p2 + F.interpolate(self.conv2(p3), size=p2.shape[2:], mode="nearest")
        p1 = p1 + F.interpolate(self.conv1(p2), size=p1.shape[2:], mode="nearest")

        return p1 

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

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
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
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
        
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

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

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class IPDFRNet(nn.Module):
    def __init__(self, opt,num_classes=1000,depths=[3, 4, 6, 3]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.sam_model,_=sam_model_registry[opt.vit_name](image_size=opt.img_size,
                                                                num_classes=opt.num_classes,
                                                                checkpoint=opt.ckpt, pixel_mean=[0, 0, 0],\
                                                                pixel_std=[1, 1, 1])
        
        self.pkg = import_module(opt.module)
        self.net = self.pkg.LoRA_Sam(self.sam_model, opt.rank).cuda()

        self.pvt_channels =[64, 128, 320, 512] #b3,4,5
        
        # 用于将 mask 的通道从 1 映射到对应 stage 的通道数
        self.f_proj1 = nn.Conv2d(1, self.pvt_channels[0], kernel_size=1)  # -> 64
        self.f_proj2 = nn.Conv2d(1, self.pvt_channels[1], kernel_size=1)  # -> 128
        self.f_proj3 = nn.Conv2d(1, self.pvt_channels[2], kernel_size=1)  # -> 320
        

        
        self.pvtv2_v2 = pvt_v2_b5()  # 64, 128, 320, 512

        save_model_v2 = torch.load('basemodel/segment_anything/modeling/pvt_v2_b5.pth')
        model_dict_v2 = self.pvtv2_v2.state_dict()
        state_dict_v2 = {k: v for k, v in save_model_v2.items() if k in model_dict_v2.keys()}
        model_dict_v2.update(state_dict_v2)
        self.pvtv2_v2.load_state_dict(model_dict_v2)

        
        self.fpn1 = FPNFusion(pvt_channels=self.pvt_channels)
        self.final_conv = nn.Conv2d(self.pvt_channels[0], 1, kernel_size=1)  # 将通道数变为 1
        self.ipn=ImplicitPriorNet(in_channels=3)

    def forward(self, x):

        # pvt分支
        f = self.net(x,multimask_output=1)
        out_sam=F.interpolate(f, size=(352, 352), mode='bilinear', align_corners=False)  # (B, 1, H1, W1)352 352

        f1 = F.interpolate(f, size=(88, 88), mode='bilinear', align_corners=False)  # (B, 1, H1, W1)# 352 88 88
        f1 = self.f_proj1(f1)  # (B, 64, H1, W1)
        f2 = F.interpolate(f, size=(44, 44), mode='bilinear', align_corners=False)  # (B, 1, H1, W1)# 352 44 44 
        f2 = self.f_proj2(f2)  # (B, 64, H1, W1)
        f3 = F.interpolate(f, size=(22, 22), mode='bilinear', align_corners=False)  # (B, 1, H1, W1)# 352 22 22
        f3 = self.f_proj3(f3)  # (B, 64, H1, W1)
        x=self.ipn(x,out_sam)


        outs= self.pvtv2_v2(x, f1, f2, f3)
        outs_Fuse = self.fpn1(outs)
        logits = self.final_conv(outs_Fuse)  # -> [B, 1, 84, 84]
        preds = F.interpolate(logits, size=(352, 352), mode='bilinear', align_corners=False)# 352 352
        return preds, out_sam

      

class PyramidVisionTransformerImpr(nn.Module):
    def __init__(self, img_size=352, patch_size=16, in_chans=3, num_classes=1, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])



        self.fbm = nn.ModuleList([
            FrequenceBandModulation(in_channels=embed_dims[0]),
            FrequenceBandModulation(in_channels=embed_dims[1]),
            FrequenceBandModulation(in_channels=embed_dims[2]),
        ])

        self.slm = SpacialLoclization(embed_dims[0])
        self.slm = SpacialLoclization(embed_dims[1])
        self.slm = SpacialLoclization(embed_dims[2])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
    
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

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = 1
    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward(self, x,f1,f2,f3):
        
        B = x.shape[0]
        outs = []    
        x1=x

        # pvt分支
        # stage 1
        x1, H, W = self.patch_embed1(x1)
        for i, blk in enumerate(self.block1):
            x1 = blk(x1, H, W)
        x1 = self.norm1(x1)
        x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 2

        xf1,f1p= self.fbm[0](x1, f1)
        xs1=self.slm(x1, f1p)
        x2=xf1+x1+xs1

        x2, H, W = self.patch_embed2(x2)
        for i, blk in enumerate(self.block2):
            x2 = blk(x2, H, W)

        x2 = self.norm2(x2)
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x2)

        # stage 3

        xf2,f2p= self.fbm[1](x2, f2)
        xs2=self.slm(xf2, f2p)
        x3=xf2+x2+xs2

        x3, H, W = self.patch_embed3(x3)
        for i, blk in enumerate(self.block3):
            x3 = blk(x3, H, W)
        x3 = self.norm3(x3)
        x3 = x3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x3)

        # stage 4
        xf3,f3p = self.fbm[2](x3, f3)
        xs3=self.slm(xf3, f3p)
        x4=xf3+x3+xs3

        x4, H, W = self.patch_embed4(x4)
        for i, blk in enumerate(self.block4):
            x4 = blk(x4, H, W)
        x4 = self.norm4(x4)
        x4 = x4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x4)

        return outs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class pvt_v2_b0(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b1(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

class pvt_v2_b2(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b3(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b4(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b5(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
