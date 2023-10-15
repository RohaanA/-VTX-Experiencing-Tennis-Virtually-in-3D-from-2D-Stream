# ============================================================
# ===                __     _________  __                  ===
# ===                \ \   / /_   _\ \/ /                  ===
# ===                 \ \ / /  | |  \  /                   ===
# ===                  \ V /   | |  /  \                   ===
# ===                   \_/    |_| /_/\_\                  ===
# ============================================================
# Authors:               Muhammad Rohaan Atique (20I-0410)
#                        Ahmed Moiz (20I-2603)
#                        Marrium Jilani (20K-1748)
# 
# Instructions:          Please read the README.md file for instructions
# ============================================================
#Models
from detect_court import start_courtdetection
from start_tracknet import start_tracknet
from detect_bounce import start_detectbounce
from detect_velocity import start_detectvelocity
from utils.tennis_utils import get_video_properties, diff_xy, remove_outliers, interpolation, get_frames

import argparse
import cv2

import numpy as np
import pandas as pd
import os

# Parse Parameters
parser = argparse.ArgumentParser()
#Adding custom arguments (more will be added later)
parser.add_argument('--video_path', type=str, required=True, help='Path to the video')
parser.add_argument('--output_video_path', type=str, required=True, help='Path to the output video')
parser.add_argument('--bounce', type=int, required=False, default=1, help='Whether to detect bounces or not')
parser.add_argument('--velocity', type=int, required=False, default=1, help='Whether to detect bounces or not')
parser.add_argument('--b', type=int, required=False, default=1, help='Whether to detect bounces or not')
parser.add_argument('--court', type=int, required=False, default=1, help='Whether to detect bounces or not')
# Loading paramters into args
args = parser.parse_args()
input_video_path = args.video_path
output_video_path = args.output_video_path

#TrackNET Parameters
n_classes = 256
tracknet_weights_path = "models/TrackNet/tracknet_weights.1"
#Bounce classifier parameters
bounce_classifier_model = "models/BounceClassifier/bounce_classifier.pkl"

#Output Paths
bounce_output_path = 'outputs/bounce_classifier.mp4'
velocity_output_path = 'outputs/velocity.mp4'

# Setting video parameters
bounce = 1
show_velocity = 1
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps, length, v_width, v_height = get_video_properties(video)

#Printing video parameters
print('fps : {}'.format(fps))
print('output_width : {}'.format(output_width))
print('output_height : {}'.format(output_height))
total = get_frames(video) #Total frames in the video
print('total frames : {}'.format(total))

# Create output folder if it doesn't exist
if not os.path.exists('outputs'):
    os.makedirs('outputs')

#Run court detection model
coords, t, frames, last = start_courtdetection(video, v_width, v_height)
#Run TrackNet model
coords, t, output_video = start_tracknet(n_classes, tracknet_weights_path, frames, total, output_video_path, fps, output_width, output_height, coords, t, last)


# everything is done, release the video
video.release()
output_video.release()

for _ in range(3):
  x, y = diff_xy(coords)
  remove_outliers(x, y, coords)

# interpolation
coords = interpolation(coords)

# velocty 
Vx = []
Vy = []
V = []
frames = [*range(len(coords))]

for i in range(len(coords)-1):
  p1 = coords[i]
  p2 = coords[i+1]
  t1 = t[i]
  t2 = t[i+1]
  x = (p1[0]-p2[0])/(t1-t2)
  y = (p1[1]-p2[1])/(t1-t2)
  Vx.append(x)
  Vy.append(y)

for i in range(len(Vx)):
  vx = Vx[i]
  vy = Vy[i]
  v = (vx**2+vy**2)**0.5
  V.append(v)

xy = coords[:]

if bounce == 1:
  start_detectbounce(output_video_path, bounce_classifier_model, xy, V)

velocity_output_video_path = "outputs/output_velocity.mp4"
# Velocity display in the video
if show_velocity == 1:
    if bounce == 1:
      start_detectvelocity(bounce_output_path, fps, output_width, output_height, V, velocity_output_video_path)
    else: 
      start_detectvelocity(output_video_path, fps, output_width, output_height, V)