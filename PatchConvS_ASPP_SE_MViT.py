import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict
import math
from torchsummary import summary
from torch.nn.modules import ModuleList, Identity
from collections import OrderedDict


def PatchConvSASPPSE_MViT():
    model = PatchConvS_ASPP_SE_MViT(blocks=[1, 1, 1, 2], ratios=2, channels=64, dims=[64, 128, 256, 512])
    return model


class patchconv(nn.Module):
    def __init__(self, patch_size=(4, 4), in_dims=3, out_dims=16):
        super(patchconv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_dims, out_channels=out_dims, kernel_size=patch_size,
                                            stride=patch_size, padding=(0, 0), bias=False),
                                  nn.BatchNorm2d(out_dims),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Resblock(nn.Module):
    def __init__(self, in_dims):
        super(Resblock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_dims, out_channels=in_dims, kernel_size=(3, 3),
                                            stride=(1, 1), padding=(1, 1), bias=False),
                                  nn.BatchNorm2d(in_dims),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_channels=in_dims, out_channels=in_dims, kernel_size=(3, 3),
                                            stride=(1, 1), padding=(1, 1), bias=False),
                                  nn.BatchNorm2d(in_dims),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        residual = x
        x = self.conv(x)
        out = x + residual
        return out


class BasicRes(nn.Module):
    def __init__(self, num_blocks, dims, ratios, downsample=True):
        super(BasicRes, self).__init__()
        self.blocks = ModuleList()
        self.downsample = patchconv(patch_size=(2, 2), in_dims=dims, out_dims=dims*ratios) if downsample else Identity()
        for i in range(num_blocks):
            block = Resblock(in_dims=dims)
            self.blocks.append(block)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(Upsample, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_dims, out_channels=out_dims, kernel_size=(5, 5),
                                            stride=(1, 1), padding=(2, 2), bias=False),
                                  nn.BatchNorm2d(out_dims),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Dropout(nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        During training, randomly zeroes some of the elements of the input tensor with probability `p` using samples \
        from a Bernoulli distribution.

        :param p: probability of an element to be zeroed. Default: 0.5
        :param inplace: If set to ``True``, will do this operation in-place. Default: ``False``
        """
        super(Dropout, self).__init__(p=p, inplace=inplace)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        input = self.forward(input)
        return input, 0.0, 0.0


class SEblock(nn.Module):
    def __init__(self, in_dim):
        super(SEblock, self).__init__()
        squeeze = int(in_dim // 16)
        self.fc1 = nn.Conv2d(in_dim, squeeze, (1, 1))
        self.fc2 = nn.Conv2d(squeeze, in_dim, (1, 1))

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.leaky_relu_(scale)
        scale = self.fc2(scale)
        scale = torch.sigmoid(scale)
        return scale * x


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates, out_channels: int = 256) -> None:
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.LeakyReLU(negative_slope=0.2, inplace=True))
        ]

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class LinearLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias = True,
                 *args, **kwargs) -> None:
        """
            Applies a linear transformation to the input data

            :param in_features: size of each input sample
            :param out_features:  size of each output sample
            :param bias: Add bias (learnable) or not
        """
        super(LinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.bias is not None and x.dim() == 2:
            x = torch.addmm(self.bias, x, self.weight.t())
        else:
            x = x.matmul(self.weight.t())
            if self.bias is not None:
                x += self.bias
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout, bias=True):
        super(MultiHeadAttention, self).__init__()
        self.qkv_proj = LinearLayer(in_features=embed_dim, out_features=3 * embed_dim, bias=bias)

        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = LinearLayer(in_features=embed_dim, out_features=embed_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward_other(self, x: Tensor) -> Tensor:
        # [B x N x C]
        b_sz, n_patches, in_channels = x.shape

        # [B x N x C] --> [B x N x 3 x h x C]
        qkv = (
            self.qkv_proj(x)
                .reshape(b_sz, n_patches, 3, self.num_heads, -1)
        )
        # [B x N x 3 x h x C] --> [B x h x 3 x N x C]
        qkv = qkv.transpose(1, 3)

        # [B x h x 3 x N x C] --> [B x h x N x C] x 3
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        query = query * self.scaling

        # [B x h x N x C] --> [B x h x c x N]
        key = key.transpose(2, 3)

        # QK^T
        # [B x h x N x c] x [B x h x c x N] --> [B x h x N x N]
        attn = torch.matmul(query, key)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [B x h x N x N] x [B x h x N x c] --> [B x h x N x c]
        out = torch.matmul(attn, value)

        # [B x h x N x c] --> [B x N x h x c] --> [B x N x C=ch]
        out = out.transpose(1, 2).reshape(b_sz, n_patches, -1)
        out = self.out_proj(out)

        return out

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_other(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim, num_heads, attn_dropout, ffn_dropout, dropout):
        super(TransformerBlock, self).__init__()
        self.LN1 = nn.Sequential(nn.LayerNorm(embed_dim))
        self.MSAttn = MultiHeadAttention(embed_dim, num_heads=num_heads, attn_dropout=attn_dropout, bias=True)
        self.Drop1 = Dropout(p=dropout)
        self.LN2 = nn.Sequential(nn.LayerNorm(embed_dim))
        self.linear1 = LinearLayer(embed_dim, ffn_latent_dim, bias=True)
        self.act = nn.Sequential(nn.ReLU(inplace=False))
        self.Drop2 = Dropout(p=ffn_dropout)
        self.linear2 = LinearLayer(ffn_latent_dim, embed_dim,  bias=True)
        self.Drop3 = Dropout(p=dropout)

    def forward(self, x):
        res1 = x
        x = self.LN1(x)
        x = self.MSAttn(x)
        x = self.Drop1(x)
        res2 = res1 + x
        x = self.LN2(res2)
        x = self.linear1(x)
        x = self.act(x)
        x = self.Drop2(x)
        x = self.linear2(x)
        x = self.Drop3(x)
        x = res2 + x
        return x


class standard_down_conv(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(standard_down_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_dims, out_channels=out_dims, kernel_size=(3, 3),
                                            stride=(2, 2), padding=(1, 1), bias=False),
                                  nn.BatchNorm2d(out_dims),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.resconv = Resblock(out_dims)

    def forward(self, x):
        x = self.conv(x)
        x = self.resconv(x)
        return x


class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, n_transformer_blocks, ffn_latent_dim, patch_w, patch_h,
                 num_heads=4, attn_dropout=0.0, ffn_dropout=0.0, dropout=0.1, fusion=True):
        super(MobileViTBlock, self).__init__()
        # self.conv5X5in = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=(5, 5),
        #                                          stride=(1, 1), padding=(2, 2), bias=False),
        #                                nn.BatchNorm2d(in_channels),
        #                                nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # self.conv1X1in = nn.Sequential(nn.Conv2d(in_channels, embed_dim, kernel_size=(1, 1),
        #                                          stride=(1, 1), padding=(0, 0), bias=False),
        #                                nn.BatchNorm2d(embed_dim),
        #                                nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.global_rep = nn.ModuleList()
        for i in range(n_transformer_blocks):
            global_rep = TransformerBlock(in_channels, ffn_latent_dim, num_heads, attn_dropout, ffn_dropout, dropout)
            self.global_rep.append(global_rep)
        # self.conv1X1out = nn.Sequential(nn.Conv2d(embed_dim, in_channels, kernel_size=(1, 1),
        #                                           stride=(1, 1), padding=(0, 0), bias=False),
        #                                 nn.BatchNorm2d(in_channels),
        #                                 nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv5X5out = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=(5, 5),
                                                  stride=(1, 1), padding=(2, 2), bias=False),
                                        nn.BatchNorm2d(in_channels),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.fusion = fusion
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.patch_area = patch_h * patch_w

    def unfolding(self, feature_map: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w # n_w
        num_patch_h = new_h // patch_h # n_h
        num_patches = num_patch_h * num_patch_w # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h
        }

        return patches, info_dict

    def folding(self, patches: Tensor, info_dict: Dict) -> Tensor:
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1)

        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            feature_map = F.interpolate(feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False)
        return feature_map

    def forward(self, x: Tensor) -> Tensor:
        res = x
        # x = self.conv5X5in(x)
        # fm = self.conv1X1in(x)
        # convert feature map to patches
        patches, info_dict = self.unfolding(x)
        # learn global representations
        for i, gr in enumerate(self.global_rep):
            patches = gr(patches)
        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)

        # fm = self.conv1X1out(fm)
        if self.fusion:
            fm = self.conv5X5out(torch.cat((res, fm), dim=1))
        return fm


class PatchConvS_ASPP_SE_MViT(nn.Module):
    def __init__(self, blocks, dims, ratios, channels):
        super(PatchConvS_ASPP_SE_MViT, self).__init__()
        self.patch_conv = patchconv(patch_size=(2, 2), in_dims=3, out_dims=channels)
        self.stages = nn.ModuleList()
        for i in range(0, len(blocks)):
            stage = BasicRes(num_blocks=blocks[i], dims=dims[i], ratios=ratios, downsample=(i + 1 < len(blocks)))
            self.stages.append(stage)
        self.ASPP = ASPP(dims[3], atrous_rates=[6, 12, 18], out_channels=dims[3])
        self.SE = SEblock(dims[3])
        self.up1 = Upsample(dims[2], dims[1])
        self.up2 = Upsample(dims[1], dims[0])
        self.up3 = Upsample(dims[0], dims[0])
        self.pwconv0 = nn.Sequential(nn.Conv2d(dims[3], dims[2], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                                     nn.BatchNorm2d(dims[2]),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.MobileViT_Block = MobileViTBlock(dims[2], n_transformer_blocks=1, ffn_latent_dim=dims[2]*3,
                                              patch_w=2, patch_h=2)
        self.pwconv = nn.Sequential(nn.Conv2d(dims[0], 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                                    nn.BatchNorm2d(1),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        concat_block = []
        x = self.patch_conv(x)
        concat_block.append(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < 3:
                concat_block.append(x)
        x = self.ASPP(x)
        x = self.SE(x)
        x = self.pwconv0(x)
        x = self.MobileViT_Block(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = x + concat_block[2]
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = x + concat_block[1]
        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = x + concat_block[0]
        x = self.up3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.pwconv(x)
        return x


class PatchConvS_ASPP_SE_MViT_Seg(nn.Module):
    def __init__(self, blocks, dims, ratios, channels, num_classes):
        super(PatchConvS_ASPP_SE_MViT_Seg, self).__init__()
        self.patch_conv = patchconv(patch_size=(2, 2), in_dims=3, out_dims=channels)
        self.stages = nn.ModuleList()
        for i in range(0, len(blocks)):
            stage = BasicRes(num_blocks=blocks[i], dims=dims[i], ratios=ratios, downsample=(i + 1 < len(blocks)))
            self.stages.append(stage)
        self.ASPP = ASPP(dims[3], atrous_rates=[6, 12, 18], out_channels=dims[3])
        self.SE = SEblock(dims[3])
        self.pwconv0 = nn.Sequential(nn.Conv2d(dims[3], dims[2], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                                     nn.BatchNorm2d(dims[2]),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.MobileViT_Block = MobileViTBlock(dims[2], n_transformer_blocks=1, ffn_latent_dim=dims[2]*3,
                                              patch_w=2, patch_h=2)
        self.classifier = nn.Sequential(nn.Conv2d(dims[2], dims[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                        nn.BatchNorm2d(dims[2]),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                        nn.Conv2d(dims[2], num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        # self.conv1 = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
        #                            nn.BatchNorm2d(num_classes),
        #                            nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        result = OrderedDict()
        x = self.patch_conv(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
        x = self.ASPP(x)
        x = self.SE(x)
        x = self.pwconv0(x)
        x = self.MobileViT_Block(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # x = self.conv1(x)
        result["out"] = x
        return result


if __name__ == '__main__':
    net = PatchConvS_ASPP_SE_MViT(blocks=[1, 1, 1, 2], ratios=2, channels=64, dims=[64, 128, 256, 512]).cpu()
    summary(net, (3, 224, 224), device='cpu')
