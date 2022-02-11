# FCRN-for-Depth-Estimation
A beginner in the CV field reruns FCRN for depth estimation...

the original paper: [Deeper depth prediction with fully convolutional residual networks](https://arxiv.org/abs/1606.00373) .

the original implementation :https://github.com/iro-cp/FCRN-DepthPrediction.git .

## Requirements
### Environment
```
python==3.6.12
torch==1.7.0
h5py==3.1.0
numpy==1.16.2
scipy==1.2.1
opencv-python==4.5.1.48
Pillow==8.0.1
```
### Datasets
**NYU depth V2**

Download the [NYU Depth Dataset V2 Labelled Dataset](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat)  and put it into  **rawdata**  folder where **split.mat** locates .

Then,
```shell
### split the dataset
$ python create_nyu_h5.py  
```

### pretrained model

Download the pretrained backbone [resnet50](https://download.pytorch.org/models/resnet50-19c8e357.pth),and put it into the project directly.

## Run
there are some parameters you can set before starting training,run ``` python main.py --h``` to check them.
### train
e.g. : ```python main.py --gpu 0 --epochs 30 --lr 1e-3```
### resume
the trained model will be saved in the **results/** ,resume the model by running:

``` python main.py --resume results/xxx/xxx.pth.tar  ```
### evaluate
you can get the average metrics , the metirc of every img and colored imgs by running:

``` python main.py --evaluate results/xxx/xxx.pth.tar  ```
