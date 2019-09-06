"""
This code revises the original source code "test.py" to test the finetuned
model on remote sensing images:
The inputs to the test are in data/remote_sensing/test/LR
The output (super resolution) goes to output/remote_sensing
The ground truth is in data/remote_sensing/test/HR
"""



import sys
import os.path
import glob
import cv2
import numpy as np
import torch
import architecture as arch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim


# import the pretrained ESRGAN
model_path = "models/RRDB_ESRGAN_remote_finetune.pth"
orig_model_path = "models/RRDB_ESRGAN_x4.pth"
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

# folder for low res and corresponding hi res image
test_LR_folder = 'input/remote_sensing/test/LR/*'
test_HR_folder = 'input/remote_sensing/test/HR/'

# setup skeleton architecture
model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
model2 = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
# load pretrained ESRGAN parameters
model.load_state_dict(torch.load(model_path), strict=True)

# switch model to testing mode
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)
print('Model path {:s}. \nTesting...'.format(model_path))
idx = 0
for path in glob.glob(test_LR_folder):
    idx += 1
    base = os.path.splitext(os.path.basename(path))[0]


    # read image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    # generate HR image
    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    # save generated image to output folder
    output_img = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output_img = (output_img * 255.0).round()
    cv2.imwrite('output/remote_sensing/{:s}_finetune.png'.format(base), output_img)

# repeat HR image generation for original model to compare results
# load pretrained ESRGAN parameters
model2.load_state_dict(torch.load(orig_model_path), strict=True)

# switch model to testing mode
model2.eval()
for k, v in model2.named_parameters():
    v.requires_grad = False
model2 = model2.to(device)
print('Model path {:s}. \nTesting...'.format(orig_model_path))
idx = 0
for path in glob.glob(test_LR_folder):
    idx += 1
    base = os.path.splitext(os.path.basename(path))[0]

    # read image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    # generate HR image
    output = model2(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    # save generated image to output folder
    output_img = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output_img = (output_img * 255.0).round()
    cv2.imwrite('output/remote_sensing/{:s}_finetune.png'.format(base), output_img)



