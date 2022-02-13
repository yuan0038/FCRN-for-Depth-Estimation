# FCRN-for-Depth-Estimation(pytorch)
A beginner in the CV  reruns FCRN for depth estimation...

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
Download [the dataset(32G)](http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz) having 47,036 training imgs and 654 test imgs then unzip it into '**Data**' folder .

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
<BR/>
## Results
I downsample and center-crop the rgb and depth images into (228,304) pixels ,and the predictions of model are upsampled to **(228,304)** pixels to calculate loss and metrics while  they are upsampled to **(480,640)** pixels in authors' paper.  [about this issue.](https://github.com/iro-cp/FCRN-DepthPrediction/issues/49)  

I train this model on Tesla T4(16GB GPU memory),and the memory it really needs is about 13GB(batch size=32).    
some results  imgs with learning rate 1e-5,5 epochs:
<img src="https://github.com/yuan0038/FCRN-for-Depth-Estimation/blob/main/results.jpg">

|     |  rms  |  rel  | delta1 | delta2 | delta3 |
|-----------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
|author|0.573|  0.127|0.811| 0.953| 0.988|
|this work|0.522|0.146|0.806|0.954|0.987|  
<BR/>  

## Citation

```
@article{Ma2017SparseToDense,
	title={Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image},
	author={Ma, Fangchang and Karaman, Sertac},
	booktitle={ICRA},
	year={2018}
}  
@article{ma2018self,
	title={Self-supervised Sparse-to-Dense: Self-supervised Depth Completion from LiDAR and Monocular Camera},
	author={Ma, Fangchang and Cavalheiro, Guilherme Venturelli and Karaman, Sertac},
	journal={arXiv preprint arXiv:1807.00275},
	year={2018}
}  
@inproceedings{laina2016deeper,
        title={Deeper depth prediction with fully convolutional residual networks},
        author={Laina, Iro and Rupprecht, Christian and Belagiannis, Vasileios and Tombari, Federico and Navab, Nassir},
        booktitle={3D Vision (3DV), 2016 Fourth International Conference on},
        pages={239--248},
        year={2016},
        organization={IEEE}
}
```


