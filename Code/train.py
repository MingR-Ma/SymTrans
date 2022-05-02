import os
import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
from DataLoad import BrainDataGenerator as DataGenerator
import argparse
from datetime import datetime, timedelta, timezone
import numpy as np
import losses
from model import SymTrans as Net
from model import SpatialTransform
from Validation import validation
import csv
from tensorboardX import SummaryWriter
import json
from tqdm import tqdm
import glob
from tools import show, save_checkpoint

warnings.filterwarnings('ignore')

test_path = '/home/mamingrui/data/OASIS_data/neast/test_neast/'

parser = argparse.ArgumentParser(description='param')
parser.add_argument('--iters', default=300001, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--checkpoint_path', default="./Checkpoint/",
                    type=str)

# Validation set
parser.add_argument('--atlas_file', default="../data/validation_data/atlases/", type=str)  # OAS2_0019
parser.add_argument('--atlas_label', default="../data/validation_data/atlases_label/", type=str)
parser.add_argument('--valsets_file', default='../data/validation_data/valsets/', type=str)
parser.add_argument('--valsets_label', default='../data/validation_data/valsets_label/', type=str)

# Train set
parser.add_argument('--data', default='../data/train_data/', type=str)
parser.add_argument('--log_folder', default='./Log/', type=str)
parser.add_argument('--range_flow', default=1.0, type=float)
parser.add_argument('--reg_smooth', default=0.02, type=float)

#Switch registration mode: diffeomorphic or dispalcement learning
parser.add_argument('--learning_mode', default='displacement', type=str)

# Validate the train after the N=2000 iterations
parser.add_argument('--VI', default=100, type=int)

args = parser.parse_args()

torch.backends.cudnn.benchmark = True


def Train(iters, lr, atlas_file, atlas_label, valsets, valsets_label,
          labels, checkpoint_path, range_flow, log_folder, reg_smooth,
          validation_iters, learning_mode):
    print(f"LOG_FLODER : {log_folder}")
    print(f"CHECKPOINT FOLDER : {checkpoint_path}")

    # atlas for validation
    atlases = sorted(glob.glob(atlas_file + '*.npy'))
    atlases_label = sorted(glob.glob(atlas_label + '*.npy'))
    print(f'Atlases :\n {atlases}')

    # valsets&label for validation
    valsets = sorted(glob.glob(valsets + '*.npy'))
    valsets_label = sorted(glob.glob(valsets_label + '*.npy'))
    print(f'Validation :\n {valsets}')

    # tensorboardX
    writer = SummaryWriter(log_folder)

    # init model
    print("Train model : {}".format(Net))

    model = Net(
        feature_shape=feature_shape, base_channel=base_channel, down_ratio=down_ratio, vit_depth=vit_depth,
        patch_size=patch_size, n_heads=n_heads, sr_ratio=sr_ratio, diff=learning_mode
    ).cuda()

    transform = SpatialTransform([96, 112, 96]).cuda()  # grid sample in my code.

    train_set = DataGenerator(trainset)

    trainset_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=0,
                                 pin_memory=True, drop_last=False)

    print("----------------Train phase----------------.")

    loss_sim = losses.MSE().loss

    Grad_loss = losses.smoothloss

    opt = Adam(model.parameters(), lr=lr)

    counter = 0
    flag = 0
    flag_jac = 0
    flag_iter = 0
    state_iter = 0
    current_iter = state_iter + 1

    data_size = len(train_set)
    print("Data size is {}. ".format(data_size))

    acc = 0.0

    # train
    loss_iters = 0.0
    tmp_a = 0.0
    tmp_b = 0.0

    while current_iter <= iters:
        update = False

        loss_epoch = 0.0

        for X, Y in tqdm(trainset_loader):
            """
            X :fixed
            Y: atlas
            """
            X_cuda = X.unsqueeze(1).cuda()
            Y_cuda = Y.unsqueeze(1).cuda()

            flow = model(X_cuda, Y_cuda)
            pred = transform(X_cuda, flow * range_flow)
            loss_1 = loss_sim(pred, Y_cuda)
            loss_2 = Grad_loss(flow * range_flow)
            loss = loss_1 + reg_smooth * loss_2

            opt.zero_grad()
            loss.backward()
            opt.step()

            # After 400 iterations report the loss value
            if current_iter % 400 == 0:
                tmp_a = tmp_a + loss_1.item()
                tmp_b = tmp_b + loss_2.item()

                loss_iters = loss_iters + loss.item()
                writer.add_scalars(f'similarity loss', {f'similarity loss': tmp_a / 400}, current_iter)
                writer.add_scalars('smooth', {'smooth_loss': tmp_b / 400}, current_iter)
                writer.add_scalars('epoch_loss', {'epoch_loss': loss_iters / 400}, current_iter)
                writer.add_image('flow_map', 100 * flow.squeeze(0)[:, :, 56, :])
                writer.close()
                loss_iters = 0.0
                tmp_a = 0.0
                tmp_b = 0.0

            if current_iter % validation_iters == 0 or current_iter - state_iter == 1:
                acc, val_time, atlas_slice, volume_slice, pred_slice, volume_label_slice, jac_det_neg_per, atlas_label_slice, pred_label_slice = \
                    validation(range_flow, atlases=atlases, atlases_label=atlases_label, valsets=valsets,
                               valsets_label=valsets_label, atlas_show=atlases[0], val_show=valsets[0],
                               model=model, labels=labels, slice=56)

                writer.add_scalars('dice score forward', {'dice_score': acc}, current_iter)

                fig = show(atlas_slice, volume_slice, pred_slice, volume_label_slice, atlas_label_slice,
                           pred_label_slice)

                flow_per = range_flow * flow.permute(0, 2, 3, 4, 1).squeeze(0)
                flow_per = flow_per.detach().cpu().numpy()
                flow_slice = np.stack([flow_per[:, 56, :, 0], flow_per[:, 56, :, 1]], axis=-1)
                print(flow_slice.shape)

                writer.add_figure('Validation', fig, current_iter)

                writer.add_scalars('jac_det negative percent', {'percent': jac_det_neg_per}, current_iter)
                writer.close()

                # scheduler.step(acc)
                print(f"Iter:{current_iter}th. Present LR:{opt.state_dict()['param_groups'][0]['lr']}.")

                if flag < acc:
                    is_best = True
                    update = 'True'
                    save_checkpoint({'iter_th': current_iter, 'loss': loss_epoch,
                                     'state_dict': model.state_dict(), 'best_acc': acc,
                                     'optimizer': opt.state_dict(), },
                                    is_best, checkpoint_path)
                    flag = acc
                    flag_jac = jac_det_neg_per
                    flag_iter = current_iter

                print(''.center(80, '='), flush=True)
                print("\t\titers: {}".format(current_iter), flush=True)
                print("\t\tLoss: {}".format(current_iter), flush=True)
                print("\t\tAccuracy (Dice score): {}.".format(acc), flush=True)
                print("\t\tValidation time spend: {:.2f}s".format(val_time), flush=True)
                print(''.center(80, '='), flush=True)

                if not os.path.exists(checkpoint_path + f'{args.reg_smooth}' + '_log.csv'):
                    with open(checkpoint_path + f'{args.reg_smooth}' + '_log.csv', 'a') as f:
                        csv_write = csv.writer(f)
                        row = ['iter_th', 'LR', 'per_epoch_time', 'loss', 'validation', 'update']
                        csv_write.writerow(row)
                else:
                    with open(checkpoint_path + f'{args.reg_smooth}' + '_log.csv', 'a') as f:
                        csv_write = csv.writer(f)
                        row = [current_iter, opt.state_dict()['param_groups'][0]['lr'], loss_epoch,
                               acc, update]
                        csv_write.writerow(row)

            if current_iter % validation_iters == 0:
                save_checkpoint({'epoch': current_iter, 'loss': loss_epoch, 'state_dict': model.state_dict(),
                                 'best_acc': flag, 'optimizer': opt.state_dict(), }, is_best=False,
                                checkpoint_path=checkpoint_path, filename=f'{current_iter}_checkpoint.pth.tar')
            current_iter += 1
            if current_iter > iters:
                save_checkpoint({'epoch': current_iter, 'loss': loss_epoch, 'state_dict': model.state_dict(),
                                 'best_acc': flag, 'optimizer': opt.state_dict(), }, is_best=False,
                                checkpoint_path=checkpoint_path, filename='final_result.pth.tar')
                torch.save(model, checkpoint_path + '/model_with_params.pth')

                params_dict = {
                    'model name': model_name,
                    'iters': flag_iter,
                    'lr': lr, 'reg_smooth': reg_smooth,
                    'range_flow': range_flow, 'best dice': flag,
                    'nonpositive Jacobian determinant percent': flag_jac,
                    'patch_size': patch_size,
                    'sr_ratio': sr_ratio,
                    'base_channel': base_channel,
                    'n_heads': n_heads
                }

                with open(f'{log_folder}/describe.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    for row in params_dict.items():
                        writer.writerow(row)

                with open(f'{checkpoint_path}describe.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    for row in params_dict.items():
                        writer.writerow(row)

                break


if __name__ == '__main__':

    labels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
              20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0]

    iters = args.iters
    lr = args.lr
    atlas_file = args.atlas_file
    atlas_label = args.atlas_label
    valsets = args.valsets_file
    valsets_label = args.valsets_label
    dataset = args.data
    range_flow = args.range_flow
    log_folder = args.log_folder
    reg_smooth = args.reg_smooth
    validation_iters = args.VI

    # Displacement or diffeomorphic registration
    learning_mode = args.learning_mode

    model_name = str(Net)
    model_name = model_name.split("'")
    model_name = model_name[1]

    image_size = (96, 112, 96)
    feature_shape = (48, 56, 48)
    base_channel = 32
    n_heads = (2, 4, 8)
    down_ratio = (2, 4, 8, 16)
    vit_depth = 2
    patch_size = (3, 3, 3)
    sr_ratio = (24, 16, 12)

    # Get the Train set
    # Using our class <dataload.py to load the nd.array data>
    trainset = glob.glob(dataset + '*.npy')

    checkpoint_path = args.checkpoint_path + f'Smooth_{reg_smooth}/' + f'range_{range_flow}/'
    log_folder = log_folder + f'Smooth_{reg_smooth}' + f'range_{range_flow}'

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        print(f"Now, this experiment's parameters [Learn Rate = {lr}]")

    Train(iters=iters, lr=lr, atlas_file=atlas_file, atlas_label=atlas_label,
          valsets=valsets, valsets_label=valsets_label, labels=labels,
          checkpoint_path=checkpoint_path, range_flow=range_flow, log_folder=log_folder,
          reg_smooth=reg_smooth, validation_iters=validation_iters, learning_mode=learning_mode)
