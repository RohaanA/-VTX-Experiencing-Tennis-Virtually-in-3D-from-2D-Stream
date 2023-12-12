from models.TrackNet.TrackNetCode import trackNet
from utils.tennis_utils import get_video_properties
import queue
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw

def start_tracknet(n_classes, tracknet_weights_path, frames, total, output_video_path, fps, output_width, output_height, coords, t, last):
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
    
    
    for img in frames:
        print('Tracking the ball(%): {}'.format(round( (currentFrame / total) * 100, 2)))
        
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
    #Store the coords in a file called ball_coords.txt
    f = open("ball_coords.txt", "w")
    for coord in coords:
        f.write(str(coord) + "\n")
    #Return the video with the ball tracked
    return coords, t, output_video
    
    