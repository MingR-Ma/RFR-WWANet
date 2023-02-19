import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
from DataLoad import UpperAbdomenDataset
import argparse
import os
import numpy as np
import losses
import RFRWWANet as mymodel
from RFRWWANet import CONFIGS
from Validation import validation
import csv
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys
from tools import show, save_checkpoint
import glob

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='param')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--n_epoch', default=300, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--loss_name', default='MSE', type=str)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--reg_param', default=0.04,
                    type=float)
parser.add_argument('--id_param', default=1, type=float)

# The path of atlas image.
parser.add_argument('--atlas_file_path',
                    default="/home/mamingrui/data/abdomen_new/before_resample_val/images/word_0001.npy",
                    type=str)
# The path of atlas label.
parser.add_argument('--atlas_label_path',
                    default="/home/mamingrui/data/abdomen_new/before_resample_val/labels/word_0001.npy", type=str)

# The folder of images for training. See Line 102.
parser.add_argument('--train_path', default='/home/mamingrui/data/abdomen_new/before_resample_train/', type=str)
# The folder of images for validation. See line 91-92.
parser.add_argument('--val_path', default='/home/mamingrui/data/abdomen_new/before_resample_val/', type=str)
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--checkpoint_path',
                    default="./",
                    type=str)
parser.add_argument('--checkpoint_file',
                    default="./abdomen/attn_0_04_RELU/checkpoints/MSE/s0.04/Best_checkpoint.pth.tar",
                    type=str)
parser.add_argument('--early_stop', default=False, type=bool)
parser.add_argument('--model', default='attn_0_04_RELU', type=str)
parser.add_argument('--width', default='norm', type=str)
parser.add_argument('--log_folder', default='./', type=str)
parser.add_argument('--init_params', default=False, type=bool)
parser.add_argument('--overwrite', default=True, type=bool)
args = parser.parse_args()

torch.backends.cudnn.benchmark = True


def Train(epochs,
          batch_size,
          loss_name,
          lr,
          reg_param,
          log_folder,
          train_path,
          atlas_file_path,
          atlas_label_path,
          val_path,
          labels,
          resume,
          checkpoint_path,
          checkpoint_file,
          approach_name,
          ):
    writer = SummaryWriter(log_folder)

    atlas = atlas_file_path
    atlas_label = atlas_label_path
    atlas_data = np.ascontiguousarray(np.load(atlas)[None, None, ...])

    print(f'Atlases :\n {atlas_file_path}')

    os.chdir(val_path)
    val_files = sorted(glob.glob(os.getcwd() + '/images/*.npy'))
    val_labels = sorted(glob.glob(os.getcwd() + '/labels/*.npy'))

    if atlas_file_path in val_files:
        val_files.remove(atlas_file_path)
    if atlas_label_path in val_labels:
        val_labels.remove(atlas_label_path)

    print(f'Validation :\n {val_files}')

    os.chdir(train_path)
    train_files = glob.glob(os.getcwd() + '/images/*.npy')

    vol_orig_shape = [192, 128, 64]

    config = CONFIGS['RFRANet']
    model = mymodel.SwinNet(config)
    model.cuda()
    print(model)

    print(loss_name)
    if loss_name == 'MSE':
        loss_fun = losses.MSE().loss
    elif loss_name == 'NCC':
        loss_fun = losses.NCC()

    else:
        raise Exception("Loss function must NCC or MSE")

    Grad_loss = losses.Grad().loss

    updated_lr = lr
    opt = Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)

    if resume:

        os.chdir(sys.path[0])

        flag = 0.0
        check_point = torch.load(checkpoint_file, map_location='cpu')
        state_iter = check_point['epoch']

        best_acc = check_point['best_acc']

        print(f'Training restart at : {state_iter}th epoch.', flush=True)

        model.load_state_dict(check_point['state_dict'])
        opt.load_state_dict(check_point['optimizer'])
        opt.param_groups[0]['lr'] = lr
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        epoch = state_iter + 1

    else:

        flag = 0
        state_iter = 1
        epoch = state_iter

        best_acc = 0

    print(opt.param_groups[0]['lr'])

    train_set = UpperAbdomenDataset(train_files)
    trainset_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                                 pin_memory=True, drop_last=False)

    y = torch.from_numpy(atlas_data).cuda()

    data_size = len(train_set)
    print("Data size is {}. ".format(data_size))

    update = False

    while epoch <= epochs:

        sum_sim_loss = []
        sum_smo_loss = []
        sum_loss = []

        print(f'Epoch: {epoch}')
        for x in tqdm(trainset_loader):
            """
            X :moving
            Y: fixed
            """
            model.train()

            x = x.cuda()

            X_Y, X_Y_flow = model(torch.cat([x, y], 1))

            loss_sim = loss_fun(X_Y, y)
            loss_smooth = Grad_loss(X_Y_flow)

            loss = loss_sim + loss_smooth * reg_param

            sum_sim_loss.append(loss_sim.item())
            sum_smo_loss.append(loss_smooth.item())
            sum_loss.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

        writer.add_scalars(f'{loss_name}_loss', {f'{loss_name}_loss': np.mean(sum_sim_loss)}, epoch)
        writer.add_scalars('smooth_loss', {'smooth_loss': np.mean(sum_smo_loss)}, epoch)
        writer.add_scalars('epoch_loss', {'epoch_loss': np.mean(sum_loss)}, epoch)
        writer.close()

        val_acc, time_spend, atlas_slice, volume_slice, pred_slice, jac_det_slice, flow, jac_neg_per = validation(
            shape=vol_orig_shape, atlases=atlas, atlases_label=atlas_label, valsets=val_files,
            valsets_label=val_labels, atlas_show=atlas, val_show=val_files[0],
            model=model, labels=labels, slice=56
        )

        fig = show(atlas_slice, volume_slice, pred_slice, jac_det_slice)

        writer.add_scalars('dice score', {'dice_score': val_acc}, epoch)
        writer.add_figure('Validation', fig, epoch)
        writer.add_scalars('jac_det negative percent', {'percent': jac_neg_per}, epoch)
        writer.close()

        print(''.center(80, '='), flush=True)
        print("\t\tLearning Rate: {}".format(opt.state_dict()['param_groups'][0]['lr']), flush=True)
        print("\t\titers: {}".format(epoch), flush=True)
        print("\t\tLoss: {}".format(np.mean(sum_sim_loss)), flush=True)
        print("\t\tAccuracy (Dice score): {}.".format(val_acc), flush=True)
        print("\t\tValidation time spend: {:.2f}s".format(time_spend), flush=True)

        print(''.center(80, '='), flush=True)

        if not os.path.exists(checkpoint_path + 'results.csv'):
            with open(checkpoint_path + 'results.csv', 'a') as f:
                csv_write = csv.writer(f)
                row = ['epoch', 'LR', 'per_epoch_time', 'loss', 'DSC', 'JAC', 'update']
                csv_write.writerow(row)
        else:
            with open(checkpoint_path + f'{args.reg_param}' + '_log.csv', 'a') as f:
                csv_write = csv.writer(f)
                row = [epoch, opt.state_dict()['param_groups'][0]['lr'], epoch,
                       val_acc, jac_neg_per, update]
                csv_write.writerow(row)

        if overwrite:
            if best_acc <= val_acc:
                save_checkpoint({'epoch': epoch, 'loss': np.mean(sum_loss), 'state_dict': model.state_dict(),
                                 'best_acc': val_acc, 'optimizer': opt.state_dict(), }, is_best=False,
                                checkpoint_path=checkpoint_path, filename=f'/Best_checkpoint.pth.tar')
                best_acc = val_acc
        else:
            save_checkpoint({'epoch': epoch, 'loss': np.mean(sum_loss), 'state_dict': model.state_dict(),
                             'best_acc': flag, 'optimizer': opt.state_dict(), }, is_best=False,
                            checkpoint_path=checkpoint_path, filename=f'/{epoch}_checkpoint.pth.tar')

        if epoch > epochs:
            break

        epoch += 1


if __name__ == '__main__':

    labels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

    epochs = args.n_epoch
    batch_size = args.batch_size
    loss_name = args.loss_name
    lr = args.lr
    reg_param = args.reg_param
    train_path = args.train_path
    atlas_file_path = args.atlas_file_path
    atlas_label_path = args.atlas_label_path
    val_path = args.val_path
    resume = args.resume
    checkpoint_file = args.checkpoint_file

    overwrite = args.overwrite

    approach_name = args.model
    log_folder = args.log_folder
    checkpoint_path = args.checkpoint_path

    log_folder = log_folder + f'abdomen/{approach_name}/log/{loss_name}/' + f's{reg_param}/'
    log_folder = os.path.abspath(log_folder)

    checkpoint_path = checkpoint_path + f'abdomen/{approach_name}/checkpoints/{loss_name}/' + f's{reg_param}/'
    checkpoint_path = os.path.abspath(checkpoint_path)

    print(f'Log_folder: {log_folder}')
    print(f'Checkpoints_folder: {checkpoint_path}')

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        print(f"Now, this experiment's parameters [Learn Rate = {lr}] [Regression para = {reg_param}] ")

    Train(
        epochs=epochs, batch_size=batch_size, loss_name=loss_name, lr=lr,
        reg_param=reg_param, log_folder=log_folder,
        train_path=train_path, atlas_file_path=atlas_file_path,
        atlas_label_path=atlas_label_path,
        val_path=val_path, labels=labels, resume=resume,
        checkpoint_path=checkpoint_path,
        checkpoint_file=checkpoint_file,
        approach_name=approach_name
    )
