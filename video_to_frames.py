# code to generate frames of a video
import cv2
import os
from os import path

input_video_folder = 'input_video'
video_names = [video for video in os.listdir(input_video_folder)]

vidcap = cv2.VideoCapture(os.path.join(input_video_folder, video_names[0]))
success,image = vidcap.read()


input_frames_folder = 'input_frames'
if not path.exists(input_frames_folder):
    os.mkdir(input_frames_folder)

count = 0
while success:
    n = len(str(count))
    suffix = '0'*(7-n) + str(count)
    cv2.imwrite(input_frames_folder + "/frame%s.jpg" % suffix, image)     
    success,image = vidcap.read()
    count += 1
    print(count)