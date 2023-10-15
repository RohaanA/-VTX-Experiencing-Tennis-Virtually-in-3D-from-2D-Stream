from models.court_detector.CourtDetector import CourtDetector
import time
import cv2

# Court lines are only detected for one frame
# The lines are then duplicated for the rest of the frames
def start_courtdetection(video, v_width, v_height):
    # court
    court_detector = CourtDetector()
    coords = []
    frame_i = 0
    frames = []
    t = []
    last = time.time() # start counting 
    
    # Detect court lines in the first frame
    ret, frame = video.read()
    if ret:
        print('Detecting the court lines...')
        lines = court_detector.detect(frame)
    else:
        return coords, t, frames, last
    
    # Process the frames
    while True:
        frame_i += 1

        if ret:
            for i in range(0, len(lines), 4):
                x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
                cv2.line(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 5)
                print("Line drawn on frame: ", frame_i, " from point: ", (int(x1),int(y1)), " to point: ", (int(x2),int(y2)))
                # Calculate the intersection point of the two lines
                x3, y3 = (x1 + x2)/2, (y1 + y2)/2
                
                # Draw a red circle at the intersection point
                cv2.circle(frame, (int(x3), int(y3)), 5, (255, 0, 0), -1)
                print("Circle drawn on frame: ", frame_i, " at point: ", (int(x3), int(y3)))
            
            new_frame = cv2.resize(frame, (v_width, v_height))
            frames.append(new_frame)
        else:
            break
        
        # Read the next frame
        ret, frame = video.read()
    
    print('Court Detection Complete.')
    # Return the frames with the court lines
    return coords, t, frames, last