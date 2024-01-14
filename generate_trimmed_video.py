# code to generate trimmed video from input frames by using a threshold value on bayesian frames 
from os import path
import cv2
import os
import numpy as np
bayesian_frames_folder = 'bayesian_output/video1'
input_frames_folder = 'input_frames'

trimmed_video_folder = 'trimmed_video'

if not path.exists(trimmed_video_folder):
      os.mkdir(trimmed_video_folder)

b_frames_array = [frame for frame in os.listdir(bayesian_frames_folder) if frame != "Thumbs.db"]
b_frames_array.sort()

i_frames_array = [frame for frame in os.listdir(input_frames_folder) if frame != "Thumbs.db"]
i_frames_array.sort()

(height, width, layers) = cv2.imread(os.path.join(bayesian_frames_folder, b_frames_array[0])).shape
writer = cv2.VideoWriter(os.path.join(trimmed_video_folder, 'trimmed_video.mp4'), 0, 30, (width, height))

threshold = 0.05      # change as per the video

for i in range(len(b_frames_array)):
    image = cv2.imread(os.path.join(bayesian_frames_folder, b_frames_array[i]))
    if np.sum(image) >= threshold*height*width*layers*255:
      input_image = cv2.imread(os.path.join(input_frames_folder, i_frames_array[i]))
      writer.write(input_image)
cv2.destroyAllWindows()
writer.release()