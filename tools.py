import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import shutil
import os
import pystrum.pynd.ndutils as nd
import torch.nn.functional as F
import glob
import sys


def save_checkpoint(state, is_best, checkpoint_path, filename='checkpoint.pth.tar'):
    best_val = []
    best_val.append(state['best_acc'])
    torch.save(state, checkpoint_path + filename)
    if is_best:
        shutil.copyfile(checkpoint_path + filename,
                        checkpoint_path + 'model_best.pth.tar')
        print('\tAccuracy is updated and the params is saved in [model_best.pth.tar]!'.ljust(20), flush=True)


def show(atlas, img, pred, jac_det):
    fig, ax = plt.subplots(1, 4)
    fig.dpi = 200

    ax0 = ax[0].imshow(atlas, cmap='gray')
    ax[0].set_title('atlas')
    ax[0].axis('off')
    cb0 = fig.colorbar(ax0, ax=ax[0], shrink=0.2)
    cb0.ax.tick_params(labelsize='small')

    ax1 = ax[1].imshow(img, cmap='gray')
    ax[1].set_title('moving')
    ax[1].axis('off')
    cb1 = fig.colorbar(ax1, ax=ax[1], shrink=0.2)
    cb1.ax.tick_params(labelsize='small')

    ax2 = ax[2].imshow(pred, cmap='gray')
    ax[2].set_title('pred')
    ax[2].axis('off')
    cb2 = fig.colorbar(ax2, ax=ax[2], shrink=0.2)
    cb2.ax.tick_params(labelsize='small')

    ax3 = ax[3].imshow(jac_det, cmap='bwr', norm=MidpointNormalize(midpoint=1))
    ax[3].set_title('jac_det')
    ax[3].axis('off')
    cb3 = fig.colorbar(ax3, ax=ax[3], shrink=0.2)
    cb3.ax.tick_params(labelsize='small')
    return fig


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def MinMaxNorm(img, use_gpu=False):
    if use_gpu:
        Max = torch.max(img)
        Min = torch.min(img)
        return (img - Min) / (Max - Min)
    else:
        Max = img.max()
        Min = img.min()
        return (img - Min) / (Max - Min)


def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    J = np.gradient(disp + grid)

    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def grid2contour(grid):
    '''
    grid--image_grid used to show deform field
    type: numpy ndarray, shape： (h, w, 2), value range：(-1, 1)
    '''
    x = np.arange(0, 96, 1)
    y = np.arange(0, 96, 1)
    X, Y = np.meshgrid(x, y)
    Z1 = grid[:, :, 0] + X

    Z2 = grid[:, :, 1] + Y

    fig = plt.figure()
    plt.contour(Z1, Y, X, 50, colors='darkgoldenrod')
    plt.contour(X, Z2, Y, 50, colors='darkgoldenrod')
    plt.xticks(()), plt.yticks(())
    plt.title('deform field')
    return fig

