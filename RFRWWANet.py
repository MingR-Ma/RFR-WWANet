# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
import torch.nn as nn
import configs
import torch.nn.functional as nnf
from Moudules import *
from einops.layers.torch import Rearrange
from einops import rearrange
import torchvision


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)

    def forward(self, src, flow):

        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""

        _, _, H, W, T = x.size()
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        if T % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - T % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x


#
# class AttnWindowAttn(nn.Module):
#     def __init__(self, dim, factor=0.25):
#         """
#         """
#         super(AttnWindowAttn, self).__init__()
#         self.factor=factor
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * self.factor),
#             nn.ReLU(),
#             nn.Linear(dim * self.factor, dim)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         """
#
#         :param x: [B L C]
#         :return:
#         """
#
#         y = torch.mean(x, dim=1, keepdim=True)
#         y = self.mlp(y)
#         y = self.sigmoid(y)
#         y = y.expand_as(x)
#         attn_window = y * x
#
#         return attn_window


class HeadTrans(nn.Sequential):
    def __init__(self, shape):
        super(HeadTrans, self).__init__()
        self.shape = shape
        self.patch_emb = nn.Sequential(
            nn.Conv3d(2, 16, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 32, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 96, 3, 2, 1),
            nn.LeakyReLU(0.2),
            Rearrange('B C D H W -> B (D H W) C'),
            nn.LayerNorm(96),
            Rearrange('B (D H W) C -> B C D H W', D=shape[0] // 4, H=shape[1] // 4, W=shape[2] // 4)
        )

    def forward(self, x):
        y = self.patch_emb(x)
        return y


class SwinNet(nn.Module):
    def __init__(self, config):
        '''
        TransMorph Model
        '''
        super(SwinNet, self).__init__()

        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim

        self.transformer = SwinTransformerHigh(patch_size=config.patch_size,
                                               in_chans=config.in_chans,
                                               embed_dim=config.embed_dim,
                                               depths=config.depths,
                                               num_heads=config.num_heads,
                                               window_size=config.window_size,
                                               mlp_ratio=config.mlp_ratio,
                                               qkv_bias=config.qkv_bias,
                                               drop_rate=config.drop_rate,
                                               drop_path_rate=config.drop_path_rate,
                                               ape=config.ape,
                                               spe=config.spe,
                                               patch_norm=config.patch_norm,
                                               use_checkpoint=config.use_checkpoint,
                                               out_indices=config.out_indices,
                                               pat_merg_rf=config.pat_merg_rf,
                                               )

        self.transformer.patch_embed = nn.Sequential()

        self.transformer.patch_embed.add_module('new_patch_emb', HeadTrans(config.img_size))

        self.up0 = DecoderBlock(embed_dim * 8, embed_dim * 4, skip_channels=embed_dim * 4 if if_transskip else 0,
                                use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0,
                                use_batchnorm=False)
        self.up2 = DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)
        self.up3 = DecoderBlock(embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if if_convskip else 0,
                                use_batchnorm=False)
        self.up4 = DecoderBlock(embed_dim // 2, config.reg_head_chan,
                                skip_channels=config.reg_head_chan if if_convskip else 0,
                                use_batchnorm=False)
        self.c1 = Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )

        self.spatial_trans = SpatialTransformer(config.img_size)

    def forward(self, x):
        source = x[:, 0:1, :, :]

        out_feats = self.transformer(x)

        f1 = out_feats[-2]
        f2 = out_feats[-3]
        f3 = out_feats[-4]
        f4 = out_feats[-5]
        f5 = out_feats[-6]

        x = self.up0(out_feats[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        x = self.up4(x, f5)
        flow = self.reg_head(x)
        out = self.spatial_trans(source, flow)
        return out, flow


class PatchExpanding(nn.Module):
    def __init__(self, dim, scale_factor=8, bias=False):
        """
        Expand operation in decoder.
        :param dim: input token channels for expanding.
        :param scale_factor: the expanding scale for token channels.
        """
        super(PatchExpanding, self).__init__()
        self.dim = dim

        self.expander = nn.Sequential(
            Rearrange('B C D H W -> B D H W C'),
            nn.Linear(self.dim, scale_factor * self.dim, bias=bias),
            Rearrange('b D H W (h d w c) -> b (D d) (H h) (W w) c', d=2, h=2, w=2, c=self.dim // 2),

        )
        self.norm = nn.Sequential(
            nn.LayerNorm(self.dim // 2, eps=1e-6),
            Rearrange('B D H W C -> B C D H W')
        )

    def forward(self, x):
        """
        Run forward pass.
        :param x: torch.Tensor
            Input tokens.
        :return: x: torch.Tensor
            Output expanded tokens.
        """
        y = self.norm(self.expander(x))

        return y


CONFIGS = {
    'RFRANet': configs.RFRANet()}


