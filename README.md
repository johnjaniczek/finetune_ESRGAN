Project Team Members:
Srilalitha Bhamidipati and John Janiczek

Selected Topic:
Media Restoration
Image and Video Restoration with Single Image Super Resolution

Contributions:
Team members worked together side by side for majority of work (50-50). Major contributions were:
1. Reproducing original results (equal)
2. Demonstrating issues of ESRGAN with remote sensing images (equal)
3. Demonstrating issues of ESRGAN with videos (equal)
4. Create the perceptual loss function from vgg19 feature stages (equal)
5. Create a pipeline for finetuning with remote sensing images (equal)
6. Create a pipeline for generating super resolution videos and finetuning (John)
7. Create presentation slide deck for background on ESRGAN (Srilalitha)
8. Create presentation slide deck for results of finetuning (John)
________________________________________________________________________________________________________________________________________
References and Citations:
Wang, Xintao, et al. "Esrgan: Enhanced super-resolution generative adversarial networks." Proceedings of the European Conference on Computer Vision (ECCV). 2018.

Link to original code (provided by Wang et. al.):
https://github.com/xinntao/ESRGAN
________________________________________________________________________________________________________________________________________
Links to datasets (4 sources of data):
1) ESRGAN code and provided test images
source code: https://github.com/xinntao/ESRGAN

2) Remote Sensing Super Resolution Dataset
srtoolbox, remote sensing image dataset: https://www.vision.uji.es/srtoolbox/download/datasets.zip

3) Video finetuning dataset:
training and testing dataset was compiled from a subset of videos from videvo sample videos.
Due to the size of the video data set (3GB) this folder was not included
but can be downloaded from dropbox:
https://www.dropbox.com/s/qagzn3lx9q9l1sn/video.zip?dl=0
***original source : https://www.videvo.net/stock-video-footage/sample/ (note, source does not compile videos into a directory)

4) PRIM2018 Test Dataset (Competition dataset ESRGAN named winner)
download from: https://pirm.github.io/

5) PRIM2018 evaluation code (Matlab code NOT needed to generate SR images, but used by PRIM2018 evaluation of perceptual index)
Note: Matlab code from competion was okayed by Professor Karam
download from: https://github.com/roimehrez/PIRM2018
also need to clone Ma score to PIRM2018/utils per evaluation code readme: https://github.com/chaoma99/sr-metric

One small video subfolder, test270 (few 2 videos, 7.7MB) is included so that some results can be reproduced
without downloading the entire video dataset


________________________________________________________________________________________________________________________________________
Listing of sub folders

-BhamidipatiJaniczek_Project (Main Project Folder)
	-ESRGAN (Original Source Code)
		-*finetune_video4.py (revised code for finetuning the ESRGAN model to perform better on video frames)
		-*finetune_remote.py (revised code for finetuning the ESRGAN model to perform better on remote sensing images)
		-*test_video_finetune.py (revised code for generating super resolution videos from the fine tuned model)
		-*display_side_by_side.py (code for displaying videos side by side for visual comparison)
		-*test_remote_sensing_ESRGAN.py (adapted code demonstrating issues of ESRGAN with remote sensing data)
		-*test_video_ESRGAN.py (adapted code demonstrating issues of ESRGAN with video data)
		-*architecture.py (architecture for ESRGAN model)
		-*block.py (building blocks for model architecture)
		-*net_interp.py (script for demonstrating interpolation)

		-figures (plots/images used to support analysis in published paper)
		-models (directory to place pre-trained models)
		-output (directory where results are stored)
			-remote_sensing (sub folder for remote_sensing results)
			-video (sub folder for video results)
			-side_by_side (sub folder for videos side by side for comparison)
		-input (directory with data that is used as input)
			-LR (low resolution images provided with original source code to reproduce main results)
			-remote_sensing (folder for remote sensing dataset from srtoolbox)
				-HR (ground truth, high resolution x4 image quality)
				-LR (model input, low resolution images with 1/4 image quality)
			-video (folder for video dataset, many videos not included because of large size, can be downloaded from dropbox)
				- test270 (included, folder of 270p videos used as model input for generating super resolution videos)
				- test360 (NOT included, folder of 360p videos used as model input for generating super resolution videos)
				- train (NOT included, 540p videos used for finetuning, no overlap with test vids)
				- train2 (NOT included, original sized videos used for finetuning, no overlap with test vids
________________________________________________________________________________________________________________________________________

Transfer learning, improvements, new features

The improvements were for two tasks finetuning the model create more realistic remote sensing images and more realistic super resolution
videos. Each task had it's own challenges and features that needed to be addressed. Note that the provided code only had a trained model
and a script which loaded the pre trained model and used it, so we needed to recreate all aspects of training from scratch given the 
descriptions in the paper

Remote Sensing Image Finetuning:
	1. Create a perceptual feature extractor by importing pretrained vgg19 model and peeling off layers
	2. Create a perceptual loss function by comparing extracted features from ground truth and generated images
	3. Create a fine tuning routine by enabling gradients on all model parameters, initializing and adam optimizer, and
	using the perceptual loss to backpropogate and optimize over several epochs.
	4. Test on an unseen data set to validate the improved image quality over ESRGAN

Super Resolution Video Finetuning:
	1. All improvements from remote sensing finetuning also needed (feature extractor, perceptual loss function, and finetuning routine)
	2. Data pipeline to import videos one frame at a time and feed into the super resolution generator
	3. Data augmentation scheme (random frame grabbing and random cropping) to avoid overfitting and decrease training time
	4. Ability to generate and save super resolution videos in a readable form (using unseen test videos)
	5. Ability to combine videos side by side and overlay text for clarity in comparing super resolution videos
________________________________________________________________________________________________________________________________________

Motivation behind revised implementations

Remote sensing super resolution: Personally, remote sensing is related to some of our thesis research work, so it made sense to try to
fit remote sensing into the project. Also, remote sensing is a field where super resolution could be very useful because the relatively
low spatiel resolution given the distance of the image results in poor image quality before compression. Super resolution images could
be useful for creating sharper remote sensing images to make it easier for humans to analyze and possibly could be easier for training
other Deep Learning algorithms to analyze and classify the earth/other planets remotely

Super resolution video finetuning: Videos take up a large ammount of space and are often compressed due to storage and communication
bandwith limits. Methods for enhancing resolution of compressed videos could be very useful for video streaming services such as 
Youtube, Netflix, or even teleconference software. The super resolution videos also have interesting challenges because of artifacts
from compression and motion blur that can be more challenging that still frame images that ESRGAN does well on.
________________________________________________________________________________________________________________________________________
Instructions to finetune ESRGAN for generating more realistic remote sensing images

### dependencies: Python3, numpy, torch (version >= 0.4), torchvision, opencv, 
Note: if wanting to skip to testing (no finetuning, go to step 6)
 
1. Ensure remote sensing super resolution dataset is downloaded and placed in correct folder per above 

2. Run fine tuning code, finetune_remote.py . The script will use a training subset to finetune and then generate
super resolution images using a never before seen test subset.

3. Results are in './results/remote_sensing' folder

4. Compare finetuned results with ESRGAN images (in same folder) and notice how much sharper
the finetuned images are.

5. Finetuned model is saved as './models/RRDB_ESRGAN_remote_finetune.pth'

### Test saved model

6. Run script test_remote_finetune.py which loads the saved model and generates HR images

7. View outputs in './output/remote_sensing' and compare ESRGAN (original model generated images) with finetune (finetuned model generated images)
________________________________________________________________________________________________________________________________________

Instructions to finetune ESRGAN for generating more realistic videos

### dependencies: Python3, numpy, torch (version >= 0.4), torchvision, opencv, moviepy (moviepy only needed for generating side by side videos, which isn't required to test code)


1. If wanting to run the finetuning code, make sure to download the video training data set from:
//www.dropbox.com/s/qagzn3lx9q9l1sn/video.zip?dl=0

(if no desire to finetune, skip to test section)

2. Run the finetuning script, finetune_video4.py
	-The script will open all videos in train2 and generate super resolution images 1 at a time
	-The script grabs random frames from each video and crops random low resolution sections
	-in order to mitigate overfitting and reduce training time

4. Resulting finetuned model is saved in ./models

### Test

5. Ensure test images are present in input/video/test270 (or whichever folder you want to test)

6. Run the test_video_finetune.py script. The script with generate super resolution videos one frame at a time

7. The results are in ./output/video. Compare with results generated from ESRGAN (should already be in folder
or can be re-generated with test_video_ESRGAN.py)

### Display side by side

8. In order to see the results more clearly run display_side_by_side.py . The script does not handle
all videos in the folder, and must be manually tweaked each time you want to generate a new side by side video
The tweaks are to simply change the name of the video files you want to place side by side and to change
the text overlay








