import torch
import argparse
import numpy as np
from model220421 import SymTrans as Net
from model220421 import SpatialTransform
from tools import jacobian_determinant
from losses import dice

parser = argparse.ArgumentParser()
parser.add_argument("--Diff_model_path", type=str, default='../Weights/Diffeormorphic/Diffeomorphic.pth')
parser.add_argument("--Dis_model_path", type=str, default='../Weights/Displacement/296000_checkpoint.pth')
parser.add_argument("--fixed", type=str, default='../data/test_data/atlas/OASIS_OAS1_0299_MR1.npy')
parser.add_argument("--moving", type=str, default='../data/test_data/valset/OASIS_OAS1_0363_MR1.npy')
parser.add_argument("--fixed_label", type=str, default='../data/test_data/atlas_label/OASIS_OAS1_0299_MR1_label.npy')
parser.add_argument("--moving_label", type=str, default='../data/test_data/valset_label/OASIS_OAS1_0363_MR1_label.npy')
parser.add_argument('--Result_dir', default='../data/Result/', type=str)
parser.add_argument('--range_flow', default=1, type=int)
parser.add_argument('--vol_shape', nargs='+', type=int, default=[96, 112, 96])
parser.add_argument('--evaluation', default=True)

#Switch registration mode: diffeomorphic or dispalcement learning
parser.add_argument('--learning_mode', default='displacement')

args = parser.parse_args()


def test(A, B, learning_mode, model_path, vol_orig_shape, range_flow, feature_shape, base_channel, n_heads, down_ratio,
             vit_depth, patch_size, sr_ratio, result_dir):

    model = Net(
        feature_shape=feature_shape, base_channel=base_channel, down_ratio=down_ratio, vit_depth=vit_depth,
        patch_size=patch_size, n_heads=n_heads, sr_ratio=sr_ratio, learning_mode=learning_mode
    ).cuda()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()


    transform = SpatialTransform(vol_orig_shape).cuda()

    A = torch.Tensor(np.load(A)).unsqueeze(0).unsqueeze(0).cuda() #fixed
    B = torch.Tensor(np.load(B)).unsqueeze(0).unsqueeze(0).cuda() #moving

    flow = model(B, A)


    warped_BA = transform(B, flow* range_flow)

    warped_BA = warped_BA.squeeze(0).squeeze(0).detach().cpu().numpy()

    np.save(result_dir + 'warped_B', warped_BA)

    if evaluation:
        return flow


if __name__ == '__main__':
    vol_orig_shape = args.vol_shape
    result_dir = args.Result_dir
    A = args.fixed
    B = args.moving
    evaluation = args.evaluation
    learning_mode = args.learning_mode
    range_flow = args.range_flow
    feature_shape = (48, 56, 48)
    base_channel = 32
    n_heads = (2, 4, 8)
    down_ratio = (2, 4, 8, 16)
    vit_depth = 2
    patch_size = (3, 3, 3)
    sr_ratio = (24, 16, 12)

    if learning_mode=='diffeomorphic':
        model_path = args.Diff_model_path
    else:
        model_path = args.Dis_model_path

    output = test(A, B, learning_mode, model_path, vol_orig_shape, range_flow, feature_shape, base_channel, n_heads, down_ratio,
                  vit_depth, patch_size, sr_ratio, result_dir)
    A_label = args.fixed_label
    A_label = np.load(A_label)
    B_label = args.moving_label
    B_label_tensor = torch.Tensor(np.load(B_label)).unsqueeze(0).unsqueeze(0).cuda()

    labels = list(np.unique(A_label))[1:]
    flow_BA = output
    transform = SpatialTransform(vol_orig_shape).cuda()
    warped_B_label = transform(B_label_tensor, flow_BA*range_flow, 'nearest')
    warped_B_label = warped_B_label.squeeze(0).squeeze(0).detach().cpu().numpy()
    dice_score = np.sum(dice(warped_B_label, A_label, labels)) / len(labels)
    np.save(result_dir + 'warped_B_label', warped_B_label)
    n_jac_det = np.sum(
        jacobian_determinant(range_flow * flow_BA.permute(0, 2, 3, 4, 1).squeeze(0).detach().cpu()) <= 0)

    flow_BA = range_flow*flow_BA.permute(0,2,3,4,1).squeeze(0).detach().cpu().numpy()
    np.save(result_dir + 'warped_BA_flow', flow_BA)

    print(f"The warp DSC score is {dice_score} with {n_jac_det} Jacobian determinants.")
