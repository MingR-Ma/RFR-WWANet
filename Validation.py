import torch
import time
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
import torch
import torch.nn.functional as nnf
import pystrum.pynd.ndutils as nd
from tqdm import tqdm
from tools import jacobian_determinant
import torch.nn as nn


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

def validation(shape,atlases, atlases_label, valsets, valsets_label, atlas_show,val_show, model,
               labels, slice=None):
    """

    :param grid:
    :param atlas: list 4
    :param atlas_label: list 4
    :param valsets: list 20
    :param image_show: atlas 1st
    :param model:
    :param labels: the selected labels to regist
    :param range_flow:
    :param slice:
    :return:
    """

    start_time = time.time()
    vol_length = len(valsets)
    atlas_length = len(atlases)
    print("Validation:")
    with torch.no_grad():

        STN = SpatialTransformer(shape,'nearest').cuda()
        model.eval()
        val_acc = []
        jac_acc = []

        atlas_volume = np.load(atlases)
        atlas_label = np.load(atlases_label)
        atlas_tensor = torch.Tensor(atlas_volume).unsqueeze(0).unsqueeze(0).cuda()

        for val, val_label in zip(valsets, valsets_label):

            val_volume = np.load(val)
            val_label = np.load(val_label)
            val_volume_tensor = torch.Tensor(val_volume).unsqueeze(0).unsqueeze(0).cuda()
            val_label_tensor = torch.Tensor(val_label).unsqueeze(0).unsqueeze(0).cuda()
            pred_volume, flow = model(torch.cat([val_volume_tensor, atlas_tensor],1))


            pred_label = STN(val_label_tensor, flow)
            pred_label = pred_label.squeeze(0).squeeze(0).detach().cpu().numpy()

            acc = dice(atlas_label, pred_label, labels)
            acc = np.mean(acc)
            val_acc.append(acc)

            flow_per = flow.permute(0, 2, 3, 4, 1)
            flow_per = flow_per.squeeze(0)
            flow_per = flow_per.detach().cpu()
            jac_det = jacobian_determinant(flow_per)

            jac_neg_per = np.sum([i <= 0 for i in jac_det]) / (jac_det.shape[0] * jac_det.shape[1] * jac_det.shape[2])
            jac_acc.append(jac_neg_per)

            atlas_slice = atlas_volume[:,:, slice]

            volume_slice = val_volume[:, :, slice]

            pred_slice = pred_volume[0, 0, :, :,slice]
            pred_slice = pred_slice.squeeze(0).squeeze(0).detach().cpu().numpy()

            jac_det_slice = jac_det[:, slice, :]

    jac_neg_per = np.mean(jac_acc)
    val_acc = np.mean(val_acc)
    time_spend = time.time() - start_time
    return val_acc, time_spend, atlas_slice, volume_slice, pred_slice, jac_det_slice, flow, jac_neg_per


def show(atlas, img, pred, jac_det):
    num = 5
    fig, ax = plt.subplots(1, num)
    fig.dpi = 150

    ax0 = ax[0].imshow(atlas, cmap='gray')
    fig.colorbar(ax0, ax=ax[0], shrink=0.3)

    ax1 = ax[1].imshow(img, cmap='gray')
    fig.colorbar(ax1, ax=ax[1], shrink=0.3)

    ax2 = ax[2].imshow(pred, cmap='gray')
    fig.colorbar(ax2, ax=ax[2], shrink=0.3)

    ax3 = ax[3].imshow(jac_det, cmap='bwr', norm=MidpointNormalize(midpoint=1))
    fig.colorbar(ax3, ax=ax[3], shrink=0.3)

    return fig


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):


        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def dice(array1, array2, labels):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    """
    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)
        dicem[idx] = top / bottom
    return dicem