import os

import torch.optim as optim

from functools import partial
from argparse import ArgumentParser

from unet.unet import UNet2D
from unet.model import Model
from unet.utils import MetricList
from unet.metrics import jaccard_index, f1_score, LogNLLLoss
from unet.dataset import JointTransform2D, ImageToImage2D, Image2D

train_dt = '/data/caozheng/caries/dataset/train_set'
val_dt = '/data/caozheng/caries/dataset/valid_set'
checkpoint_path = '/data/caozheng/caries/exp'
device = 'cpu'
in_channels = 1
out_channels = 2
depth = 5
width = 32
epochs =100
batch_size = 8
save_freq = 10
save_model = 0
model_name = 'model'
lr_rate = 1e-3
crop = None

tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)

train_dataset = ImageToImage2D(train_dt, tf_val)
val_dataset = ImageToImage2D(val_dt, tf_val)
predict_dataset = Image2D(val_dt)

conv_depths = [int(width*(2**k)) for k in range(depth)]
unet = UNet2D(in_channels, out_channels, conv_depths)
loss = LogNLLLoss()
optimizer = optim.Adam(unet.parameters(), lr=lr_rate)

results_folder = os.path.join(checkpoint_path, model_name)

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

metric_list = MetricList({'jaccard': partial(jaccard_index),
                          'f1': partial(f1_score)})

model = Model(unet, loss, optimizer, results_folder, device=device)

model.fit_dataset(train_dataset, n_epochs=epochs, n_batch=batch_size,
                  shuffle=True, val_dataset=val_dataset, save_freq=save_freq,
                  save_model=save_model, predict_dataset=predict_dataset,
                  metric_list=metric_list, verbose=True)
