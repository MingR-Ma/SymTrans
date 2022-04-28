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


def show(atlas, img, pred, img_label, atlas_label_slice, pred_label_slice):
    fig, ax = plt.subplots(2, 3)
    fig.dpi = 200

    ax0 = ax[0][0].imshow(atlas, cmap='gray')
    ax[0][0].set_title('atlas')
    ax[0][0].axis('off')

    ax1 = ax[0][1].imshow(img, cmap='gray')
    ax[0][1].set_title('moving')
    ax[0][1].axis('off')

    ax2 = ax[0][2].imshow(pred, cmap='gray')
    ax[0][2].set_title('pred')
    ax[0][2].axis('off')

    ax4 = ax[1][0].imshow(atlas_label_slice, cmap='tab20')
    ax[1][0].set_title('atlas_label')
    ax[1][0].axis('off')

    ax4 = ax[1][1].imshow(img_label, cmap='tab20')
    ax[1][1].set_title('moving_label')
    ax[1][1].axis('off')

    ax5 = ax[1][2].imshow(pred_label_slice, cmap='tab20')
    ax[1][2].set_title('pred_label')
    ax[1][2].axis('off')

    return fig


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


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

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


