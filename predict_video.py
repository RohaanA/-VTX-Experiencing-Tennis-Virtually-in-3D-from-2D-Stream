# ============================================================
#                    __     _________  __
#                    \ \   / /_   _\ \/ /
#                     \ \ / /  | |  \  / 
#                      \ V /   | |  /  \ 
#                       \_/    |_| /_/\_\
# ============================================================
# Authors:               Muhammad Rohaan Atique (20I-0410)
#                        Ahmed Moiz (20I-2603)
#                        Marrium Jilani (20K-1748)
# 
# Instructions:          Please read the README.md file for instructions
# ============================================================
from PIL import Image, ImageDraw
from utils.tennis_utils import get_video_properties
from models.TrackNet.TrackNetCode import trackNet
import argparse
import cv2
import imutils
import queue
import time
import numpy as np


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
tracknet_weights_path = "model.1"

# get video fps&video size
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

video = cv2.VideoCapture(input_video_path)
coords = []
frame_i = 0
frames = []
t = []
last = time.time() # start counting 
# while (True):
for img in frames:
    print('Tracking the ball: {}'.format(round( (currentFrame / total) * 100, 2)))
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