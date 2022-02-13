import csv
import os
import shutil
import cv2
import torch
from ParseArgument import  *
import numpy as np

from PIL import Image





def merge_into_row(input, depth_target, depth_pred):
    rgb = np.transpose(np.squeeze(input.cpu().numpy()),(1,2,0))  # h, w, c
    rgb=np.asarray(rgb).astype('uint8')
    depth_target,depth_target_col = Depth2Colored(depth_target)
    depth_pred, depth_pred_col = Depth2Colored(depth_pred)

    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge

def add_row(img_merge, row):
    return np.vstack([img_merge, row])

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def _is_pil_image(img):

        return isinstance(img, Image.Image)

def _is_tensor_image(img):
    return torch.is_tensor(img)
def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)
def Depth2Colored(img):
    if _is_tensor_image(img)and img.ndim == 4:
        img = img[0].data.squeeze().cpu().numpy().astype(np.float32)
    elif  _is_numpy_image(img):
        img=img
    else:

        raise TypeError('img should be 2d ndarray or 4d tensor. Got {} {}'.format(img.ndim,type(img)))

    if isinstance(img, np.ndarray):
        # handle numpy array
        d_min = np.min(img)
        d_max = np.max(img)
        depth_relative = (img - d_min) / (d_max - d_min)
        img=(depth_relative*255).astype('uint8')
        heat_img=cv2.applyColorMap(img,colormap=cv2.COLORMAP_JET)
        heat_img=cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
        return img,heat_img

def get_output_directory():

    output_directory = os.path.join('results',
                                    'FCRN.gpu={}.loss_fn={}.bs={}.lr={}.wd={}.'
                                    .format(args.gpu,args.loss, args.batch_size,args.lr,args.weight_decay))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    return output_directory

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)

    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
        print("best model saves as {} ".format(checkpoint_filename))

    # if epoch > 0:
    #     prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch-1) + '.pth.tar')
    #     if os.path.exists(prev_checkpoint_filename):
    #         os.remove(prev_checkpoint_filename)
class Val_imgs_metrics_Saver():
    def __init__(self,output_directory,fieldnames):
        csv_name='val_imgs_metric.csv'
        self.saver_csv=os.path.join(output_directory,csv_name)
        print("{} created".format(csv_name))
        self.fieldnames=fieldnames
        with open(self.saver_csv, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
    def __call__(self, results):
        with open(self.saver_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow({'mse': results.mse, 'rmse': results.rmse, 'absrel': results.absrel, 'lg10': results.lg10,
                             'mae': results.mae, 'delta1': results.delta1, 'delta2': results.delta2, 'delta3': results.delta3,
                             })

class Progrecess_bar():
    def __init__(self,datalen,title='progress'):
        self.datalen=datalen
        self.title=title
    def __call__(self, i):
        percent = i / self.datalen
        if percent > 1:
            percent = 1
        res = int(50 * percent) * '#'
        print('\r%s: [%-50s] %d%%' % (self.title,res, int(100 * percent)), end='')


