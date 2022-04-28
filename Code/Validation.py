import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from tools import jacobian_determinant
from model import SpatialTransform
from losses import dice


def validation(range_flow, atlases, atlases_label, valsets, valsets_label, atlas_show,val_show, model,
               labels, slice=None):

    start_time = time.time()
    vol_length = len(valsets)
    print(vol_length)
    atlas_length = len(atlases)
    print("Validation:")
    print(atlas_length, vol_length,len(labels))

    val_acc_sum = 0.0
    # val_acc_sum_=0.0
    jac_acc_sum = 0.0

    atlases=sorted(atlases)
    atlases_label=sorted(atlases_label)
    valsets=sorted(valsets)
    valsets_label=sorted(valsets_label)

    with torch.no_grad():

        STN = SpatialTransform([96,112,96]).cuda()
        model.eval()

        for atlas, atlas_label in zip(atlases, atlases_label):

            atlas_volume = np.load(atlas)

            atlas_label = np.load(atlas_label)
            atlas_tensor = torch.Tensor(atlas_volume).unsqueeze(0).unsqueeze(0).cuda()

            acc_list=[]
            jac_list=[]

            for val, val_label in zip(valsets, valsets_label):

                val_volume = np.load(val)

                val_label = np.load(val_label)
                val_volume_tensor = torch.Tensor(val_volume).unsqueeze(0).unsqueeze(0).cuda()
                val_label_tensor = torch.Tensor(val_label).unsqueeze(0).unsqueeze(0).cuda()

                flow=model(val_volume_tensor,atlas_tensor)

                pred = STN(val_volume_tensor, flow * range_flow)
                pred_label=STN(val_label_tensor,flow*range_flow,'nearest')
                pred_label = pred_label.squeeze(0).squeeze(0).detach().cpu().numpy()

                flow=(flow*range_flow).permute(0,2,3,4,1)
                acc = np.sum(dice(atlas_label, pred_label, labels))/len(labels)

                acc_list.append(acc)

                flow_per = flow.squeeze(0)
                flow_per = flow_per.detach().cpu()
                jac_det = jacobian_determinant(flow_per)

                jac_neg_per = np.sum([i <= 0 for i in jac_det]) / (
                        jac_det.shape[0] * jac_det.shape[1] * jac_det.shape[2])
                jac_list.append(jac_neg_per)

                if atlas == atlas_show and val == val_show:
                    atlas_slice = atlas_volume[:, slice, :]

                    volume_slice = val_volume[:, slice, :]
                    volume_label_slice=val_label[:,slice,:]

                    pred_slice = pred[0, 0, :, slice, :]
                    pred_slice = pred_slice.squeeze(0).squeeze(0).detach().cpu().numpy()

                    jac_det_slice = jac_det[:, slice, :]
                    atlas_label_slice=atlas_label[:,slice,:]
                    pred_label_slice=pred_label[:,slice,:]

            print(f"A atlas Accuracy dice : {acc_list}.")
            val_acc=np.sum(acc_list)/vol_length

            print(f"Average predict for atlas dice is {val_acc}")
            jac_neg_per=np.sum(jac_list)/vol_length

            val_acc_sum=val_acc+val_acc_sum

            jac_acc_sum=jac_neg_per+jac_acc_sum

    jac_neg_per = jac_acc_sum/atlas_length
    val_acc = val_acc_sum / atlas_length

    time_spend = time.time() - start_time
    return val_acc, time_spend, atlas_slice, volume_slice, pred_slice, volume_label_slice, jac_neg_per,atlas_label_slice,pred_label_slice



