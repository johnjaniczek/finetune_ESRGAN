Project Team Members:
Srilalitha Bhamidipati and John Janiczek

________________________________________________________________________________________________________________________________
References and Citations:
Wang, Xintao, et al. "Esrgan: Enhanced super-resolution generative adversarial networks." Proceedings of the European Conference on Computer Vision (ECCV). 2018.

Link to original code (provided by Wang et. al.):
https://github.com/xinntao/ESRGAN
________________________________________________________________________________________________________________________________
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


________________________________________________________________________________________________________________________________
Listing of sub folders
```bash
finetune_ESRGAN/ 		# main project folder with scripts to finetune and test models
├── input			# sub-directory for input data
│   ├── remote_sensing		# remote sensing input data
│   │   ├── test		# test-split of remote sensing input data, high and low resolution
│   │   │   ├── HR		
│   │   │   └── LR		
│   │   └── train		# train-split of remote sensing input data, high and low resolution
│   │       ├── HR		
│   │       └── LR		
│   └── video			# video input data (need to download, see link)
├── models			# sub-directory for trained models
├── output			# model results/ouputs are stored here
│   ├── remote_sensing		# remote sensing results			
│   │   └── finetune_epochs	# results with varying number of finetuning epochs
│   │       ├── 20epochs
│   │       ├── 40epochs
│   │       └── 60epochs
│   ├── side_by_side		# input and output videos played side by side
│   └── video			# video results
└── __pycache__	
```



________________________________________________________________________________________________________________________________

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
________________________________________________________________________________________________________________________________
Motivation behind revised implementations

Remote sensing super resolution: Personally, remote sensing is related to some of our thesis research work, so it made sense to try to fit remote sensing into the project. Also, remote sensing is a field where super resolution could be very useful because the relatively low spatiel resolution given the distance of the image results in poor image quality before compression. Super resolution images could be useful for creating sharper remote sensing images.

Super resolution video finetuning: Videos take up a large ammount of space and are often compressed due to storage and communication bandwith limits. Methods for enhancing resolution of compressed videos could be very useful for video streaming services such as Youtube, Netflix, or even teleconference software. The super resolution videos also have interesting challenges because of artifacts from compression and motion blur that can be more challenging that still frame images.
________________________________________________________________________________________________________________________________

### Instructions to finetune ESRGAN for generating more realistic remote sensing images

dependencies: Python3, numpy, torch (version >= 0.4), torchvision, opencv, 
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

### Instructions to finetune ESRGAN for generating more realistic videos

dependencies: Python3, numpy, torch (version >= 0.4), torchvision, opencv, moviepy (moviepy only needed for generating side by side videos, which isn't required to test code)


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








