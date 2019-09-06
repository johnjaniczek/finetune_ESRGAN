"""
Video Super Resolution Fine Tuning 1:
This code revises the original source code by finetuning the original
model to produce realistic super resolution videos

This finetune algorithm imports an existing super resolution algorithm
(finetunes ESRGAN, or continues finetuning another model)
and fine tunes by producing a super resolution (x4) image for each
frame of the video. The perceptual loss between in super resolution
and the ground truth frame is calculated and propagated back through
the network via gradient descent and the Adam optimizer.

The major contributions of this revision are:
1. Build a pipeline and add features to generate super resolution videos
by processing one frame of a video at a time
2. Build transformation tools to randomly select video frames and
random cropped locations to avoid overfitting and to speed up the
training process
3. Enable gradients on the model
4. Initialize an Adam optimizer (optimizer not provided in source code, nor mentioned in paper)
5. Create a feature extractor from VGG19 (mentioned in ESRGAN paper but not provided in code)
6. Use feature extractor to create a perceptual loss
7. Iterate over a training data set and optimize the model to reduce the perceptual loss
8. Test model by generating super resolution images using an unseen dataset (see test_video_finetune.py)

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
lr = 1e-5
epochs = 10
crop_height = 320
crop_width = 640
frames_per_epoch = 100

# folder of training input,
# y (target): original video from folder
# x (input) = video downsampled by factor of 4
train_vid_folder = 'input/video/train2/*'

# initialize pretrained ESRGAN model
model_path = "models/RRDB_ESRGAN_vid_finetune4.pth"
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
model.load_state_dict(torch.load(model_path), strict=True)
print('Model path {:s}. \nFinetuning ESRGAN...'.format(model_path))

# switch model to train mode
model.train()
for k, v in model.named_parameters():
    v.requires_grad = True
model = model.to(device)

# initialize the feature extractor vgg19 model
loss_model = torchvision.models.vgg19(pretrained=True).cuda()
vgg19_54 = nn.Sequential(*list(loss_model.features.children())[:9])
vgg19_22 = nn.Sequential(*list(loss_model.features.children())[:3])

# initialize criterion and optimizer
criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Random crop transform

ToPIL = torchvision.transforms.ToPILImage()
ToTensor= torchvision.transforms.ToTensor()
RandCrop = torchvision.transforms.RandomCrop((crop_height, crop_width))

TargetTransform = torchvision.transforms.Compose([ToPIL,
                                           RandCrop,
                                           ToTensor])
# Training input transform down samples by 4
DownSamp4 = torchvision.transforms.Resize((int(crop_height/4), int(crop_width/4)))
TrainTransform = torchvision.transforms.Compose([ToPIL,
                                                DownSamp4,
                                                ToTensor])

# open up all video captures
video_captures = [cv2.VideoCapture(path) for path in glob.glob(train_vid_folder)]

# iterate over input set in epochs
for epoch in range(epochs):
    running_loss = 0
    for idx in range(frames_per_epoch):
        # iterate over each video in training folder
        for i, cap in enumerate(video_captures):
            # reset gradient
            optimizer.zero_grad()

            # select a random frame from video
            RandFrame = np.random.randint(0, cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, RandFrame)

            # read a frame of video
            ret, img = cap.read()

            # if the video returned a frame, continue
            if ret == True:

                # pre process frame per expected dimensions
                img = img * 1.0 / 255
                img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()

                # transform target image to randomly crop a segment of frame
                img = TargetTransform(img)

                # create training input (reduce resolution by 4)
                train_img = TrainTransform(img)

                # unsqueeze input so it looks like a batch
                img = img.unsqueeze(0)
                train_img = train_img.unsqueeze(0)
                img = img.to(device)
                train_img = train_img.to(device)

                # compute output of ESRGAN
                output = model(train_img)

                # extract features of generated image
                out_feats = vgg19_22(output)

                # extract features of target image
                target_feats = vgg19_22(img)

                # compute loss between model output and target features (perceptual loss)
                loss = criterion(out_feats, target_feats)
                running_loss += loss.item()

                # perform gradient descent
                loss.backward()
                optimizer.step()



    # Release everything if job is finished
    print("epoch: %d" % epoch, "loss: %5f" % running_loss)

# save model and end
torch.save(model.state_dict(), "models/RRDB_ESRGAN_vid_finetune4.pth")
cv2.destroyAllWindows()

# release capture for next video
for i, cap in enumerate(video_captures):
    cap.release()
