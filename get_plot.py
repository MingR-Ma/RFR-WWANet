import torch
import torch.nn as nn
import numpy as np
import losses

import torch.nn.functional as F
import csv
import pystrum.pynd.ndutils as nd
from torch.autograd import Variable
from math import exp
import torch.nn.functional as nnf
import os
import pandas as pd
import glob
import RFRWWANet as TransMorph
from RFRWWANet import CONFIGS as CONFIGS_TM
# import hausdorff


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


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


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


# 275.pth
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
atlas_path = "/home/mamingrui/data/abdomen_new/before_resample_val/images/word_0001.npy"
atlas_label_path = "/home/mamingrui/data/abdomen_new/before_resample_val/labels/word_0001.npy"
train_path = "/home/mamingrui/data/abdomen_new/before_resample_train/"

# val_path="/home/mamingrui/data/BTCV/before_resample_train/affine/"
val_path = "/home/mamingrui/data/abdomen_new/before_resample_val/"
os.chdir(val_path)
val_files = sorted(glob.glob(os.getcwd() + '/images/*.npy'))[1:]
val_labels = sorted(glob.glob(os.getcwd() + '/labels/*.npy'))[1:]
csv_path = 'Validation_dice_csv/'

atlas = atlas_path
atlas_label = atlas_label_path
atlas_data = np.ascontiguousarray(np.load(atlas)[None, None, ...])
# atlas_label_data = np.ascontiguousarray(np.load(atlas_label))
print(f'Atlases :\n {atlas_path}')

# val_files & label


os.chdir(os.path.dirname(os.path.realpath(__file__)))

checkpoint = "/home/mamingrui/PycharmProjects/Thoracic_cavity_and_abdominal_cavity/Code4CVPRSubmission/abdomen/attn_0_04_RELU/checkpoints/MSE/s0.04/Best_checkpoint.pth.tar"

atlas = torch.from_numpy(np.ascontiguousarray(np.load(atlas_path)[None, None, ...])).cuda()

atlas_label = np.load(atlas_label_path)

config = CONFIGS_TM['RFRANet']
model = TransMorph.SwinNet(config)
model.cuda()
print(model)

check_point = torch.load(checkpoint, map_location='cpu')
model.load_state_dict(check_point['state_dict'])
model.eval()
SSIM = SSIM3D()

STN = SpatialTransformer((192, 128, 64), 'nearest').cuda()

label_name = ['Left-Cerebral-W. Matter',
              'Left-Cerebral-Cortex',
              'Left-Lateral-Ventricle',
              'Left-Inf-Lat-Ventricle',
              'Left-Cerebellum-W. Matter',
              'Left-Cerebellum-Cortex',
              'Left-Thalamus'
              ]

if not os.path.exists(csv_path):
    os.makedirs(csv_path)
df = pd.DataFrame(columns=['atlas_data', 'val_data', 'labels', 'label_name', 'dsc'], dtype=object)
df_ssim = pd.DataFrame(columns=['atlas_data', 'val_data', 'SSIM', 'jac'], dtype=object)
df_haus = pd.DataFrame(columns=['atlas_data', 'val_data', 'label_name', 'haus'], dtype=object)

hus = [[], [], [], [], [], [], []]
ddic = [[], [], [], [], [], [], []]
jjac = []
alldic = []
allhau = []
for i in range(len(val_files)):
    val = np.load(val_files[i])
    val_name = os.path.split(val_files[i])[1]
    print(val_name)

    val_label = np.load(val_path + "/labels/" + val_name)
    print(np.unique(val_label))
    val = torch.from_numpy(np.ascontiguousarray(val[None, None, ...])).cuda()

    val_label = torch.Tensor(val_label).unsqueeze(0).unsqueeze(0).cuda()
    val_label.float()
    X_Y, X_Y_flow = model(torch.cat([val, atlas], 1))

    labels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    pred_label = STN(val_label, X_Y_flow)
    print(np.unique(pred_label.squeeze(0).squeeze(0).detach().cpu().numpy()))

    dice_score = dice(atlas_label, pred_label.squeeze(0).squeeze(0).detach().cpu().numpy(), labels)

    spacing = (1.5, 1.5, 1.0)

    # Haus = []
    # for i in range(0, 7):
    #     al = atlas_label
    #     pre_l = pred_label.squeeze(0).squeeze(0).detach().cpu().numpy()
    #     al = np.isin(al, labels[i]).astype(np.uint8)
    #     pre_l = np.isin(pre_l, labels[i]).astype(np.uint8)
    #     distance = hausdorff.compute_surface_distances(np.array(al, dtype=bool), np.array(pre_l, dtype=bool), spacing)
    #     haus = hausdorff.compute_robust_hausdorff(distance, 95)
    #     Haus.append(haus)

    flow_per = X_Y_flow.permute(0, 2, 3, 4, 1)
    flow_per = flow_per.squeeze(0)
    flow_per = flow_per.detach().cpu().numpy()
    jac_det = jacobian_determinant(flow_per)

    jac_neg_per = np.sum([i <= 0 for i in jac_det]) / (jac_det.shape[0] * jac_det.shape[1] * jac_det.shape[2])

    jjac.append(jac_neg_per)
    alldic.append(np.mean(dice_score))
    # allhau.append(np.mean(Haus))
    for i, dsc in enumerate(dice_score):
        ddic[i].append(dsc)
        df.loc[len(df)] = ["001", val_name, labels[i], label_name[i], dsc]
    # for i, ha in enumerate(Haus):
    #     hus[i].append(ha)
    #     df_haus.loc[len(df_haus)] = ["001", val_name, labels[i], ha]

    ssim = SSIM(atlas, X_Y)

    df_ssim.loc[len(df_ssim)] = ["001", val_name, ssim.detach().cpu().numpy(), jac_neg_per]

df.to_csv(csv_path + 'dice.csv', index=False)
df_ssim.to_csv(csv_path + 'ssim.csv', index=False)
df_haus.to_csv(csv_path + 'haus.csv', index=False)
df = df.drop(index=df.index)
df_haus = df_haus.drop(index=df_haus.index)
df_ssim = df_ssim.drop(index=df_ssim.index)

for i, dsc in enumerate(ddic):
    print("dsc", i, ":mean:", np.mean(ddic[i]), "std:", np.std(ddic[i]))

for i, dsc in enumerate(hus):
    print("HD", i, ":mean:", np.mean(hus[i]), "std:", np.std(hus[i]))
print("dice:", np.mean(alldic), "  std:", np.std(alldic))
print("jac:", np.mean(jjac), "  std:", np.std(jjac))
print("HD:", np.mean(allhau), "  std:", np.std(allhau))
