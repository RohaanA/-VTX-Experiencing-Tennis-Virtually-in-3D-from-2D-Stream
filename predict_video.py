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
from PIL import Image, ImageDraw
from utils.tennis_utils import get_video_properties, diff_xy, remove_outliers, interpolation
from models.TrackNet.TrackNetCode import trackNet
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from pickle import load
from models.court_detector.CourtDetector import CourtDetector
import argparse
import cv2
import imutils
import queue
import time
import numpy as np
import pandas as pd

# parse parameters
parser = argparse.ArgumentParser()
#Adding custom arguments (more will be added later)
parser.add_argument('--video_path', type=str, required=True, help='Path to the video')
parser.add_argument('--output_video_path', type=str, required=True, help='Path to the output video')

#Default parameters
if parser.parse_args().video_path is None:
    video_path = "videos/small_sample.mp4"

args = parser.parse_args()
input_video_path = args.video_path
output_video_path = args.output_video_path

#TrackNET Parameters
n_classes = 256
tracknet_weights_path = "models/TrackNet/tracknet_weights.1"
bounce_classifier_model = "models/BounceClassifier/bounce_classifier.pkl"

# get video fps&video size
bounce = 1
show_velocity = 1
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
print('fps : {}'.format(fps))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('output_width : {}'.format(output_width))
print('output_height : {}'.format(output_height))
# try to determine the total number of frames in the video file
if imutils.is_cv2() is True :
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT
else : 
    prop = cv2.CAP_PROP_FRAME_COUNT
total = int(video.get(prop))
print('total frames : {}'.format(total))

# start from first frame
currentFrame = 0
# TrackNet params
width, height = 640, 360
img, img1, img2 = None, None, None
# load TrackNet model
modelFN = trackNet
m = modelFN(n_classes, input_height=height, input_width=width)
m.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
m.load_weights(tracknet_weights_path)
# In order to draw the trajectory of tennis, we need to save the coordinate of previous 7 frames
q = queue.deque()
for i in range(0, 8):
    q.appendleft(None)

# save prediction images as videos
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

# loop over frames from the video file stream
fps, length, v_width, v_height = get_video_properties(video)

# court
court_detector = CourtDetector()

video = cv2.VideoCapture(input_video_path)
coords = []
frame_i = 0
frames = []
t = []
last = time.time() # start counting 
# Add all images to frames
while True:
    ret, frame = video.read()
    frame_i += 1

    if ret:
        if frame_i == 1:
            print('Detecting the court and the players...')
            lines = court_detector.detect(frame)
        else: # then track it
            lines = court_detector.track_court(frame)
        
        for i in range(0, len(lines), 4):
            x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
            cv2.line(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 5)
        new_frame = cv2.resize(frame, (v_width, v_height))
        frames.append(new_frame)
    else:
        break

for img in frames:
    print('Tracking the ball(%): {}'.format(round( (currentFrame / total) * 100, 2)))
    frame_i += 1
    
    # detect the ball
    # img is the frame that TrackNet will predict the position
    # since we need to change the size and type of img, copy it to output_img
    output_img = img

    # resize it
    img = cv2.resize(img, (width, height))
    # input must be float type
    img = img.astype(np.float32)

    # since the odering of TrackNet  is 'channels_first', so we need to change the axis
    X = np.rollaxis(img, 2, 0)
    # prdict heatmap
    pr = m.predict(np.array([X]))[0]

    # since TrackNet output is ( net_output_height*model_output_width , n_classes )
    # so we need to reshape image as ( net_output_height, model_output_width , n_classes(depth) )
    pr = pr.reshape((height, width, n_classes)).argmax(axis=2)

    # cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
    pr = pr.astype(np.uint8)

    # reshape the image size as original input image
    heatmap = cv2.resize(pr, (output_width, output_height))

    # heatmap is converted into a binary image by threshold method.
    ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

    # find the circle in image with 2<=radius<=7
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                              maxRadius=7)
    
    PIL_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(PIL_image)

    # check if there have any tennis be detected
    if circles is not None:
        # if only one tennis be detected
        if len(circles) == 1:

            x = int(circles[0][0][0])
            y = int(circles[0][0][1])

            coords.append([x,y])
            t.append(time.time()-last)

            # push x,y to queue
            q.appendleft([x, y])
            # pop x,y from queue
            q.pop()

        else:
            coords.append(None)
            t.append(time.time()-last)
            # push None to queue
            q.appendleft(None)
            # pop x,y from queue
            q.pop()

    else:
        coords.append(None)
        t.append(time.time()-last)
        # push None to queue
        q.appendleft(None)
        # pop x,y from queue
        q.pop()

    # draw current frame prediction and previous 7 frames as yellow circle, total: 8 frames
    for i in range(0, 8):
        if q[i] is not None:
            draw_x = q[i][0]
            draw_y = q[i][1]
            bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
            draw = ImageDraw.Draw(PIL_image)
            draw.ellipse(bbox, outline='yellow')
            del draw

    # Convert PIL image format back to opencv image format
    opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)

    output_video.write(opencvImage)

    # next frame
    currentFrame += 1

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
  # Predicting Bounces 
  test_df = pd.DataFrame({'x': [coord[0] for coord in xy[:-1]], 'y':[coord[1] for coord in xy[:-1]], 'V': V})

  # df.shift
  for i in range(20, 0, -1): 
    test_df[f'lagX_{i}'] = test_df['x'].shift(i, fill_value=0)
  for i in range(20, 0, -1): 
    test_df[f'lagY_{i}'] = test_df['y'].shift(i, fill_value=0)
  for i in range(20, 0, -1): 
    test_df[f'lagV_{i}'] = test_df['V'].shift(i, fill_value=0)

  test_df.drop(['x', 'y', 'V'], 1, inplace=True)

  Xs = test_df[['lagX_20', 'lagX_19', 'lagX_18', 'lagX_17', 'lagX_16',
        'lagX_15', 'lagX_14', 'lagX_13', 'lagX_12', 'lagX_11', 'lagX_10',
        'lagX_9', 'lagX_8', 'lagX_7', 'lagX_6', 'lagX_5', 'lagX_4', 'lagX_3',
        'lagX_2', 'lagX_1']]
  Xs = from_2d_array_to_nested(Xs.to_numpy())

  Ys = test_df[['lagY_20', 'lagY_19', 'lagY_18', 'lagY_17',
        'lagY_16', 'lagY_15', 'lagY_14', 'lagY_13', 'lagY_12', 'lagY_11',
        'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6', 'lagY_5', 'lagY_4',
        'lagY_3', 'lagY_2', 'lagY_1']]
  Ys = from_2d_array_to_nested(Ys.to_numpy())

  Vs = test_df[['lagV_20', 'lagV_19', 'lagV_18',
        'lagV_17', 'lagV_16', 'lagV_15', 'lagV_14', 'lagV_13', 'lagV_12',
        'lagV_11', 'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6', 'lagV_5',
        'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1']]
  Vs = from_2d_array_to_nested(Vs.to_numpy())

  X = pd.concat([Xs, Ys, Vs], 1)

  # load the pre-trained classifier  
  clf = load(open(bounce_classifier_model, 'rb'))

  predcted = clf.predict(X)
  idx = list(np.where(predcted == 1)[0])
  idx = np.array(idx) - 10

  video = cv2.VideoCapture(output_video_path)

  output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(video.get(cv2.CAP_PROP_FPS))
  length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  fourcc = cv2.VideoWriter_fourcc(*'XVID')

  print(fps)
  print(length)

  output_video = cv2.VideoWriter('outputs/final_video.mp4', fourcc, fps, (output_width, output_height))
  i = 0
  while True:
    ret, frame = video.read()
    if ret:
      # if coords[i] is not None:
      if i in idx:
        center_coordinates = int(xy[i][0]), int(xy[i][1])
        radius = 3
        color = (255, 0, 0)
        thickness = -1
        cv2.circle(frame, center_coordinates, 10, color, thickness)
      i += 1
      output_video.write(frame)
    else:
      break

  video.release()
  output_video.release()

velocity_output_video_path = "outputs/output_velocity.mp4"
# Velocity display in the video
if show_velocity == 1:
    # Load the output video
    velocity_video = cv2.VideoCapture(output_video_path)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    velocity_output_video = cv2.VideoWriter(velocity_output_video_path, fourcc, fps, (output_width, output_height))

    # Loop over frames from the velocity video
    while True:
        ret, frame = velocity_video.read()
        if ret is False:
            break

        # Get the frame number
        frame_number = int(velocity_video.get(cv2.CAP_PROP_POS_FRAMES))

        # Get the corresponding velocity value
        velocity = V[frame_number]

        # Draw velocity information on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'Velocity: {velocity:.2f}'
        org = (10, 30)
        fontScale = 1
        color = (0, 255, 0)  # Green color
        thickness = 2

        # Draw a filled rectangle as the background
        rectangle_padding = 10
        text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
        rectangle_width = text_size[0] + 2 * rectangle_padding
        rectangle_height = text_size[1] + 2 * rectangle_padding
        rectangle_position = (org[0], org[1] - text_size[1] - rectangle_padding)
        cv2.rectangle(frame, rectangle_position, (rectangle_position[0] + rectangle_width, rectangle_position[1] + rectangle_height), (0, 0, 0), -1)

        # Write the velocity text on the frame
        cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

        # Write the frame to the output video
        velocity_output_video.write(frame)

    # Release the velocity video and output video
    velocity_video.release()
    velocity_output_video.release()