"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""
import os
import sys

ROOT_DIR = os.environ['ROOT_DIR']
try:
    ROOT_DIR = os.environ['ROOT_DIR']
except:
    try:
        ROOT_DIR = os.environ['PWD']
    except:
        assert ROOT_DIR, 'ROOT_DIR was not able to be defined'
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.misc import imsave
import tensorflow as tf

slim = tf.contrib.slim

# Import OSVOS files
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
import osvos
from dataset import Dataset

os.chdir(root_folder)

# User defined parameters
seq_name = "truss_complex"
gpu_id = 0
train_model = True

img_path = os.path.join('data', 'interim', 'osvos', 'JPEGImages', '480p', seq_name)
anno_path = os.path.join('data', 'interim', 'osvos', 'Annotations', '480p', seq_name)
result_path = os.path.join('data', 'processed', 'osvos', 'segmentation', '480p', seq_name)

# Output: img+mask combined
save_img_result = True
overlay_color = [0, 0, 255]
transparency = 0.6

# Train parameters
parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
logs_path = os.path.join('models', seq_name)
max_training_iters = 500

# Define Dataset
test_frames = sorted(os.listdir(img_path))
test_imgs = [os.path.join(img_path, frame) for frame in test_frames]
if train_model:
    train_imgs = [os.path.join(img_path, 'image-001.jpg') + ' ' +
                  os.path.join(anno_path, 'image-001.png')]
    dataset = Dataset(train_imgs, test_imgs, './', data_aug=True)
else:
    dataset = Dataset(None, test_imgs, './')

# Train the network
if train_model:
    # More training parameters
    learning_rate = 1e-8
    save_step = max_training_iters
    side_supervision = 3
    display_step = 10
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            osvos.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
                                 save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=seq_name)

# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        checkpoint_path = os.path.join('models', seq_name, seq_name + '.ckpt-' + str(max_training_iters))
        osvos.test(dataset, checkpoint_path, result_path,
                   save_img_result, overlay_color, transparency)

# # Save results
# overlay_color = [255, 0, 0]
# transparency = 0.6
# # plt.ion()
# for img_p in test_frames:
#     frame_num = img_p.split('.')[0]
#     img = np.array(Image.open(os.path.join(img_path, img_p)))
#     mask = np.array(Image.open(os.path.join(result_path, 'masks', frame_num + '.png')))
#     mask /= np.max(mask)
#     im_over = np.ndarray(img.shape)
#     im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (
#     overlay_color[0] * transparency + (1 - transparency) * img[:, :, 0])
#     im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (
#     overlay_color[1] * transparency + (1 - transparency) * img[:, :, 1])
#     im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (
#     overlay_color[2] * transparency + (1 - transparency) * img[:, :, 2])
#
#     imsave(os.path.join(result_path, 'images', img_p), im_over)
#
#     # plt.imshow(im_over.astype(np.uint8))
#     # plt.axis('off')
#     # plt.show()
#     # plt.pause(0.01)
#     # plt.clf()
