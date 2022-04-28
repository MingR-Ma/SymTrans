# SymTrans: Symmetric Transformer-based model for image registration


## Trained model

We uploaded the weights, including the displacement and diffeomorphic registration model's weights.

## Training

If you would like to train this model on your own dataset, conver you data to `numpy.array` (i.e. `.npy`) format, then put them in `/Data/train_data/`.
To validate the training process, put the validation data in `/Data/validation_data/`. In detail, put the atlases to `/Data/validation_data/atlases/`; put the atlases' labels to `/Data/validation_data/atlases_label/`. Correspondingly, put moving (source) images and their labels in `/Data/validation_data/valsets/` and `/Data/validation_data/valsets_label/`.

```
./scripts/tf/train.py --img-list /images/list.txt --model-dir /models/output --gpu 0
```

The `--img-prefix` and `--img-suffix` flags can be used to provide a consistent prefix or suffix to each path specified in the image list. Image-to-atlas registration can be enabled by providing an atlas file, e.g. `--atlas atlas.npz`. If you'd like to train using the original dense CVPR network (no diffeomorphism), use the `--int-steps 0` flag to specify no flow integration steps. Use the `--help` flag to inspect all of the command line options that can be used to fine-tune network architecture and training.


## Registration

If you simply want to register two images, you can use the `register.py` script with the desired model file. For example, if we have a model `model.h5` trained to register a subject (moving) to an atlas (fixed), we could run:

```
./scripts/tf/register.py --moving moving.nii.gz --fixed atlas.nii.gz --moved warped.nii.gz --model model.h5 --gpu 0
```

This will save the moved image to `warped.nii.gz`. To also save the predicted deformation field, use the `--save-warp` flag. Both npz or nifty files can be used as input/output in this script.


## Testing (measuring Dice scores)

To test the quality of a model by computing dice overlap between an atlas segmentation and warped test scan segmentations, run:

```
./scripts/tf/test.py --model model.h5 --atlas atlas.npz --scans scan01.npz scan02.npz scan03.npz --labels labels.npz
```

Just like for the training data, the atlas and test npz files include `vol` and `seg` parameters and the `labels.npz` file contains a list of corresponding anatomical labels to include in the computed dice score.


## Parameter choices


### CVPR version

For the CC loss function, we found a reg parameter of 1 to work best. For the MSE loss function, we found 0.01 to work best.


### MICCAI version

For our data, we found `image_sigma=0.01` and `prior_lambda=25` to work best.

In the original MICCAI code, the parameters were applied after the scaling of the velocity field. With the newest code, this has been "fixed", with different default parameters reflecting the change. We recommend running the updated code. However, if you'd like to run the very original MICCAI2018 mode, please use `xy` indexing and `use_miccai_int` network option, with MICCAI2018 parameters.


## Spatial Transforms and Integration

- The spatial transform code, found at `voxelmorph.layers.SpatialTransformer`, accepts N-dimensional affine and dense transforms, including linear and nearest neighbor interpolation options. Note that original development of VoxelMorph used `xy` indexing, whereas we are now emphasizing `ij` indexing.

- For the MICCAI2018 version, we integrate the velocity field using `voxelmorph.layers.VecInt`. By default we integrate using scaling and squaring, which we found efficient.



  
    **Learning Conditional Deformable Templates with Convolutional Networks**  
  [Adrian V. Dalca](http://adalca.mit.edu), [Marianne Rakic](https://mariannerakic.github.io/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
  NeurIPS 2019. [eprint arXiv:1908.02738](https://arxiv.org/abs/1908.02738)

