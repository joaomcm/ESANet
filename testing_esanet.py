import argparse

import torch
import torch.nn.functional as F
import numpy as np
from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
# from src.confusion_matrix import ConfusionMatrixTensorflow
from src.prepare_data import prepare_data
import pickle
from sens_reader import scannet_scene_reader
from matplotlib import pyplot as plt
import os

args = pickle.load(open('args.p','rb'))
model, device = build_model(args, n_classes=40)

checkpoint = torch.load(args.ckpt_path,
                        map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to(device)

lim= -1
root_dir = "/home/motion/data/scannet_v2"
my_ds = scannet_scene_reader(root_dir, 'scene0050_00',lim = lim)
tmp = my_ds[0]

dataset, preprocessor = prepare_data(args, with_input_orig=True)
import time

from tqdm import tqdm
for i in tqdm(range(0,len(my_ds),30)):
    start = time.time()
    tmp = my_ds[i]
    img_rgb = tmp['color']
    img_depth = tmp['depth']
    h, w, _ = img_rgb.shape

    # preprocess sample
    sample = preprocessor({'image': img_rgb, 'depth': img_depth})

    # add batch axis and copy to device
    image = sample['image'][None].to(device)
    depth = sample['depth'][None].to(device)

    h, w, _ = img_rgb.shape

    # preprocess sample
    sample = preprocessor({'image': img_rgb, 'depth': img_depth})

    # add batch axis and copy to device
    image = sample['image'][None].to(device)
    depth = sample['depth'][None].to(device)

    # apply network
    pred = model(image, depth)
#     print(pred.shape)
#     pred = F.interpolate(pred, (h, w),
#                          mode='bilinear', align_corners=False)
    pred = torch.argmax(pred, dim=1)
    pred = pred.cpu().numpy().squeeze().astype(np.uint8)
#     print('took {}'.format(time.time()-start))
#     # show result
    # pred_colored = dataset.color_label(pred, with_void=False)
    # fig, axs = plt.subplots(1, 3, figsize=(16, 3))
    # [ax.set_axis_off() for ax in axs.ravel()]
    # axs[0].imshow(img_rgb)
    # axs[1].imshow(img_depth, cmap='gray')
    # axs[2].imshow(pred_colored)

    # plt.suptitle(f"Image")
    # # plt.savefig('./result.jpg', dpi=150)
    # plt.show()