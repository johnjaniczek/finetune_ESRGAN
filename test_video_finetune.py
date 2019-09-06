"""
This script generates super resolution videos with the fine tuned
model using never before seen videos in the test folder

The inputs to the test are in data/videos/test360 (or test540 with small
modification to the code)
The output (super resolution) goes to output/video
"""



import sys
import os.path
import glob
import cv2
import numpy as np
import torch
import architecture as arch

# folder of test videos
test_vid_folder = 'input/video/test270/*'

# initialize pre-trained ESRGAN fine tuned for videos
model_path = "models/RRDB_ESRGAN_vid_finetune4.pth"
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
model.load_state_dict(torch.load(model_path), strict=True)

# switch to evaluate mode
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)
print('Model path {:s}. \nProcessing Video...'.format(model_path))

for path in glob.glob(test_vid_folder):
    cap = cv2.VideoCapture(path)

    # Define the codec and create VideoWriter object
    base = os.path.splitext(os.path.basename(path))[0]
    FPS = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output/video/{:s}_finetune4.avi'.format(base),
                          fourcc,
                          cap.get(cv2.CAP_PROP_FPS),
                          (int(width*4), int(height*4)))


    # process video
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            img = img * 1.0 / 255
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(device)

            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)

            # write the flipped frame
            out.write(output)

            cv2.imshow('frame', output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
