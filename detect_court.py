from models.court_detector.CourtDetector import CourtDetector
import time
import cv2
import os
import numpy as np


def get_intersection_points(frame_height, frame_width, line_coords):
    """_summary_

    Args:
        frame_height (int): _description_
        frame_width (int): _description_
        line_coords (list): _description_

    Returns:
        list of tuples: _description_
    """
    intersections = []
    for i in range(len(line_coords)):
        for j in range(i + 1, len(line_coords)):
            x1, y1, x2, y2 = map(int, line_coords[i])
            x3, y3, x4, y4 = map(int, line_coords[j])

            # Calculate the intersection point
            denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denominator != 0:
                x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
                y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
                intersections.append((int(x), int(y)))
    correct_intersections = [] 
    for point in intersections:
        if (point[0] < 0 or point[0] > frame_width or point[1] < 0 or point[1] > frame_height):
            continue
        # normalized_point = (int(point[0] / frame_width), int(point[1] / frame_height))
        correct_intersections.append(point)
    
    # Turn correct_intersections into a set to remove duplicates
    correct_intersections = list(set(correct_intersections))
    # Sort the points by x coordinate
    correct_intersections.sort(key=lambda tup: tup[0])
    # Convert to a list of tuples
    correct_intersections = [tuple(l) for l in correct_intersections]
    return correct_intersections
# Court lines are only detected for one frame
# The lines are then duplicated for the rest of the frames
def start_courtdetection(video, v_width, v_height):
    # court
    line_coords = []
    court_detector = CourtDetector()
    coords = []
    frame_i = 0
    frames = []
    t = []
    last = time.time() # start counting 
    
    
    # Detect court lines in the first frame
    ret, frame = video.read()

    # Process the frames
    while True:
        frame_i += 1

        if ret:
            if frame_i == 1:
                lines = court_detector.detect(frame)
            else: # then track it
                lines = court_detector.track_court(frame)
            frame_line_coords = []
            for i in range(0, len(lines), 4):
                x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
                cv2.line(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 5)
                # print("Line drawn on frame: ", frame_i, " from point: ", (int(x1),int(y1)), " to point: ", (int(x2),int(y2)))
                #Store the coordinates of the lines in a list
                line_coords.append([x1, y1, x2, y2])
                frame_line_coords.append([x1, y1, x2, y2])
                # Calculate the intersection point of the two lines
                # x3, y3 = (x1 + x2)/2, (y1 + y2)/2
                
                # Draw a red circle at the intersection point
                # cv2.circle(frame, (int(x3), int(y3)), 5, (255, 0, 0), -1)
                # print("Circle drawn on frame: ", frame_i, " at point: ", (int(x3), int(y3)))
            frame_intersections = get_intersection_points(v_height, v_width, frame_line_coords)
            # Append to outputs/intersections.txt
            with open('outputs/intersections.txt', 'a') as f:
                f.write("[")
                for item in frame_intersections:
                    f.write(" %s " % str(item))
                f.write("]\n")
            new_frame = cv2.resize(frame, (v_width, v_height))
            frames.append(new_frame)
            
        else:
            break
        
        # Read the next frame
        ret, frame = video.read()
    
    print('Court Detection Complete.')


    # #Draw intersections on the reference court
    # image = cv2.imread('court_point_reference.png')
    # #The intersections list contains some points outside the image, and a ton of duplicates
    # #hence we remove all points outside the image, and remove the duplicates 
    # correct_intersections = [] 
    # for point in intersections:
    #     if (point[0] < 0 or point[0] > v_width or point[1] < 0 or point[1] > v_height):
    #         continue
    #     #Normalize the coordinates of the points to the size of the reference image
    #     cv2.circle(image, point, 5, (255, 0, 0), -1)
    #     normalized_point = (int(point[0] / v_width), int(point[1] / v_height))
    #     #Also write the coordinates of the intersections and the normalized coordinates on the image, above the point
    #     cv2.putText(image, str(point), (point[0] - 50, point[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    #     cv2.putText(image, str(normalized_point), (point[0] - 50, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    #     correct_intersections.append(point)
    # cv2.imwrite('outputs/intersections.png', image)
    # #Also store all intersections in a file
    # #remove duplicates
    # correct_intersections = list(set(correct_intersections))
    # print(len(correct_intersections), " intersections found.")
    # #Sort the points by x coordinate
    # correct_intersections.sort(key=lambda tup: tup[0])
    # #Store the coordinates of the intersections in a file
    # with open('outputs/intersections.txt', 'w') as f:
    #     for item in correct_intersections:
    #         f.write("%s\n" % str(item))
    # #Lastly, draw the intersections on the frames
    # for frame in frames:
    #     for point in correct_intersections:
    #         cv2.circle(frame, point, 5, (255, 0, 0), -1)
    
    # Return the frames with the court lines
    return coords, t, frames, last