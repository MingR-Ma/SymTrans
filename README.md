# SymTrans: Symmetric Transformer-based model for image registration

## Paper:

 **Symmetric Transformer-based Network for Unsupervised Image Registration**
 The proposed SymTrans architecture：
 ![image](Architectures_paper.png)
 
Please cite: https://arxiv.org/abs/2204.13575

## Trained model

We uploaded the weights, including the displacement and diffeomorphic registration model's weights.

## Training

If you would like to train this model on your own dataset, conver you data to `numpy.array` (i.e. `.npy`) format, then put them in `/Data/train_data/`.
To validate the training process, put the validation data in `/Data/validation_data/`. In detail, put the atlases to `/Data/validation_data/atlases/`; put the atlases' labels to `/Data/validation_data/atlases_label/`. Correspondingly, put moving (source) images and their labels in `/Data/validation_data/valsets/` and `/Data/validation_data/valsets_label/`.

Excute this comand train the SymTran after allocate the dataset：

```
python train.py
```
Checkpoints and training logs, including validation results and loss values, are recorded in the`./Chekcpoint/` and './Log/' folder. You can use tensorboardx to moniter the training. Using the parameter `--learning_mode ` to select diffeomorphic or displacement registration (default `--learning_mode displacement`).

All the parameters can be found in the `train.py` and `test.py`. You can modify them if you would like to configure your own training or testing.
