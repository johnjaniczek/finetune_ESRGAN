"""
This code revises the original source code "test.py" to add additional
features for fine tuning the model to work well on remote sensing images
The major additions to the code are:

1. Enable gradients on the model
2. Initialize an Adam optimizer (optimizer not provided in source code, nor mentioned in paper)
3. Create a feature extractor from VGG19 (mentioned in ESRGAN paper but not provided in code)
4. Use feature extractor to create a perceptual loss
5. Iterate over a training data set and optimize the model to reduce the perceptual loss
6. Test model by generating super resolution images using an unseen dataset

The inputs to the finetuning (training data) are in:
    input/remote_sensing/train/LR (image)
    daa/remote_sensing/train/HR (label/target for desired super resolution image)
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


# hyper parameters
epochs = 60
lr = 1e-3

# import the pretrained ESRGAN
model_path = "models/RRDB_ESRGAN_x4.pth"
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

# folder for low res and corresponding hi res image
train_LR_folder = 'input/remote_sensing/train/LR/*'
train_HR_folder = 'input/remote_sensing/train/HR/'
test_LR_folder = 'input/remote_sensing/test/LR/*'
test_HR_folder = 'input/remote_sensing/test/HR/'

# initialize the feature extractor vgg19 model
loss_model = torchvision.models.vgg19(pretrained=True).cuda()
vgg19_54 = nn.Sequential(*list(loss_model.features.children())[:9])
vgg19_22 = nn.Sequential(*list(loss_model.features.children())[:3])


# setup skeleton architecture
model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')

# load pretrained ESRGAN parameters
model.load_state_dict(torch.load(model_path), strict=True)

# save generated images of pre-finetuned model
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)
print('Model path {:s}. \nTesting...'.format(model_path))
idx = 0

# iterate through test images
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
    cv2.imwrite('output/remote_sensing/{:s}_ESRGAN.png'.format(base), output_img)

# switch model to training mode
model.train()
for k, v in model.named_parameters():
    v.requires_grad = True
model = model.to(device)

# initialize criterion and optimizer
criterion = nn.MSELoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)


# Fine tune model
print('Model path {:s}. \nFine tuning...'.format(model_path))
idx = 0
running_loss = 0.0
for epoch in range(epochs):
    optimizer.zero_grad()
    running_loss = 0
    for path in glob.glob(train_LR_folder):
        idx += 1

        # zero the parameter gradients
        base = os.path.splitext(os.path.basename(path))[0]
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)
        output = model(img_LR)
        out_feats = vgg19_22(output)

        # extract features of target image
        target_file = train_HR_folder + "HR" + base[2:] + ".tif"
        target_img = cv2.imread(target_file, cv2.IMREAD_COLOR)
        target_img = target_img * 1.0 / 255
        target_img = torch.from_numpy(np.transpose(target_img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        target_img_LR = target_img.unsqueeze(0)
        target_img_LR = target_img_LR.to(device)
        target_feats = vgg19_22(target_img_LR)

        # compute loss between output and target
        loss = criterion(out_feats, target_feats)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        # running_loss += loss.item()


    # do one gradient step

    print("epoch: %d, loss: %f" %(epoch, running_loss))




# Test finetuned model by generating new unseen images
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = img2.max()
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


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

# save finetuned model
torch.save(model.state_dict(), "models/RRDB_ESRGAN_remote_finetune.pth")




