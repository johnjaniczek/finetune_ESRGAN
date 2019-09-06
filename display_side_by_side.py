"""
This script displays two videos side by side
It is used to demonstrate that the finetuned model generates
more realistic videos than the ESRGAN (original source code) and
that it is more realistic than the input data
"""

import os
from moviepy.editor import VideoFileClip, clips_array, vfx, TextClip, CompositeVideoClip
from moviepy.video.fx.crop import crop
file1 = "input/video/test270/setangi_beach_270p.mp4"
file2 = "output/video/setangi_beach_270p_finetune4.avi"
file1_txt = "setangi_beach_270p"
file2_txt = "finetuned"
width = 480*4
height = 270*4
clip1 = VideoFileClip(file1, target_resolution=(height, width)).margin(10)
clip2 = VideoFileClip(file2, target_resolution=(height, width)).margin(10)
clip1 = crop(clip1, x_center=width/2, y_center=height/2, width=width/4, height=height/4).margin(10)
clip2 = crop(clip2, x_center=width/2, y_center=height/2, width=width/4, height=height/4).margin(10)
txt_clip1 = TextClip(file1_txt, fontsize=36, color="white").set_position("center").set_duration(5)
txt_clip2 = TextClip(file2_txt, fontsize=36, color="white").set_position("center").set_duration(5)
final_clip = clips_array([[CompositeVideoClip([clip1, txt_clip1]),
                           CompositeVideoClip([clip2, txt_clip2])]])
final_clip.write_videofile("output/side_by_side/%s_vs_%s_crop.mp4" %(file1_txt, file2_txt))