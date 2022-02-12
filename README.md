# FCRN-for-Depth-Estimation
A beginner in the CV field reruns FCRN for depth estimation...

the original paper: [Deeper depth prediction with fully convolutional residual networks](https://arxiv.org/abs/1606.00373) .  
the original implementation of FCRN:https://github.com/iro-cp/FCRN-DepthPrediction.git .  
the orginal code of this project:[sparse-to-dense.pytorch](https://github.com/fangchangma/sparse-to-dense.pytorch) .
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
### NYU depth V2 Dataset
Download the [NYU Depth Dataset V2 Labelled Dataset](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat)  and put it into  **rawdata**  folder where **split.mat** locates .Then run:  
```shell
### split the dataset
$ python create_nyu_h5.py  
```
### pretrained model
Download the pretrained backbone [resnet50](https://download.pytorch.org/models/resnet50-19c8e357.pth),and put it into the project directly.  
<BR/>

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

## results
I downsample and center-crop the rgb and depth images into (228,304) pixels ,and the predictions of model are upsampled to (228,304) pixels to calculate loss and metrics while  they are upsampled to (480,640) pixels in authors' paper.   
If you want to get better results, you can download (the dataset)[http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz] having 47036 training imgs  to replace the Labelled Dataset(795 training imgs).  
training on 795 imgs with learning rate 1e-5,30 epochs:

|     |  rms  |  rel  | delta1 | delta2 | delta3 |
|-----------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
|author|0.573|  0.127|0.811| 0.953| 0.988|
