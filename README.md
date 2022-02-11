# FCRN-for-Depth-Estimation
A beginner in the CV field uses FCRN for Depth Estimation...

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


```
### Datasets
**NYU depth V2**

Download the [NYU Depth Dataset V2 Labelled Dataset](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat)  and put it into  **rawdata**  folder where **split.mat** locates .

Then,
```shell
### split the dataset
$ python create_nyu_h5.py  
```
