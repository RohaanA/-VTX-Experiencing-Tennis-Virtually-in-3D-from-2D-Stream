#Imports
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
# Points from the input feed (EXPERIMENTAL)
match_stream_points = np.array([
    [582, 316], [1237, 316], [582, 857], [1237, 857]
], dtype=np.float32)

# Points for the reference court image
reference_points_test = np.array([
    [287, 562], [1241, 562], [287, 2935], [1241, 2935]
], dtype=np.float32)

# Convert the points to numpy arrays
match_stream_points = np.float32(match_stream_points)

def start_inference(bounce_data, ball_coordinates, reference_points, input_video_path="videos/small_sample.mp4"):
    cap = cv2.VideoCapture(input_video_path)
    # Calculate the transformation matrix
    transform_matrix = cv2.getPerspectiveTransform(match_stream_points, reference_points)
    # Create a window for displaying the output
    cv2.namedWindow('Reference Court Image')
    i_frame = 0
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)
            player1_coordinates = None
            player2_coordinates = None
            
            for result in results:
                box = result.boxes
                #Player 1 inference only if there is a box detected
                if len(box) > 0:
                    player1 = box[0]
                    player1_coords = player1.xyxy.tolist()
                    #Print bottom mid coordinate of box player1
                    print("Player1 Coordinates: ", player1_coords)
                    player1_middle = int(player1_coords[0][0]+(player1_coords[0][2]-player1_coords[0][0])/2), int(player1_coords[0][3])
                    #Draw a dot on the bottom mid coordinate of box player1 and player2
                    cv2.circle(frame, player1_middle, 5, (0, 0, 255), -1)
                    print("Player1 Middle Coordinates: ", player1_middle)
                    player1_coordinates = np.float32([player1_middle[0], player1_middle[1], 1]).reshape(-1, 1)
                #Player 2 inference only if there is a box detected
                if len(box) > 1:
                    player2 = box[1]
                    player2_coords = player2.xyxy.tolist()
                    #Print bottom mid coordinate of box player2
                    print("Player2 Coordinates: ", player2_coords)
                    player2_middle = int(player2_coords[0][0]+(player2_coords[0][2]-player2_coords[0][0])/2), int(player2_coords[0][3])
                    #Draw a dot on the bottom mid coordinate of box player2
                    cv2.circle(frame, player2_middle, 5, (0, 0, 255), -1)
                    print("Player2 Middle Coordinates: ", player2_middle)
                    player2_coordinates = np.float32([player2_middle[0], player2_middle[1], 1]).reshape(-1, 1)
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            if (player1_coordinates is not None) and (player2_coordinates is not None):
                # Apply the transformation matrix to the player coordinates
                mapped_player1_coordinates = np.dot(transform_matrix, player1_coordinates)
                mapped_player2_coordinates = np.dot(transform_matrix, player2_coordinates)
                mapped_ball_coordinates = None
                if (ball_coordinates[i_frame] is not None):
                    mapped_ball_coordinates = np.dot(transform_matrix, np.append(ball_coordinates[i_frame], 1))
                
                # Normalize the mapped coordinates
                mapped_player1_coordinates /= mapped_player1_coordinates[2]
                mapped_player2_coordinates /= mapped_player2_coordinates[2]
                

                # Extract the x and y coordinates from the mapped coordinates
                mapped_player1_x, mapped_player1_y = mapped_player1_coordinates[0], mapped_player1_coordinates[1]
                mapped_player2_x, mapped_player2_y = mapped_player2_coordinates[0], mapped_player2_coordinates[1]

                # Read the reference court image
                reference_image = cv2.imread('court_reference.png')
                # Draw circles on the mapped player coordinates
                cv2.circle(reference_image, (int(mapped_player1_x), int(mapped_player1_y)), 15, (0, 0, 255), -1)
                cv2.circle(reference_image, (int(mapped_player2_x), int(mapped_player2_y)), 15, (0, 0, 255), -1)
                # Draw circle on the mapped ball coordinates
                if (mapped_ball_coordinates is not None):
                    if (bounce_data[i_frame] is not None):
                        # Draw an X on the ball coordinates if there is a bounce
                        line_length = 40  # Increase the line length to make the X larger
                        line_thickness = 5  # Increase the line thickness to make the X larger

                        cv2.line(reference_image, (int(mapped_ball_coordinates[0]) - line_length, int(mapped_ball_coordinates[1]) - line_length),
                                (int(mapped_ball_coordinates[0]) + line_length, int(mapped_ball_coordinates[1]) + line_length), (0, 0, 255), line_thickness)
                        cv2.line(reference_image, (int(mapped_ball_coordinates[0]) + line_length, int(mapped_ball_coordinates[1]) - line_length),
                                (int(mapped_ball_coordinates[0]) - line_length, int(mapped_ball_coordinates[1]) + line_length), (0, 0, 255), line_thickness)
                    else:
                        cv2.circle(reference_image, (int(mapped_ball_coordinates[0]), int(mapped_ball_coordinates[1])), 15, (0,255,255), -1)
                    
                # Resize the reference court image
                output_width = 300  # Specify the desired width of the output image
                output_height = int(reference_image.shape[0] * output_width / reference_image.shape[1])  # Calculate the proportional height
                output_image = cv2.resize(reference_image, (output_width, output_height))

                # Display the resized reference court image with the circles
                cv2.imshow('Reference Court Image', output_image)
                cv2.waitKey(1)
                i_frame += 1
            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", frame)

            #Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

def load_points_from_file(filename):
    points_2D = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if line != "None":
                point = line.strip("[]").split(",")
                x = int(point[0].strip())
                y = int(point[1].strip())
                points_2D.append((x, y))
            else:
                points_2D.append(None)
    return points_2D
def load_bounce_data(filename):
    bounce_data = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if line != "None":
                bounce_data.append(1)
            else:
                bounce_data.append(None)
    return bounce_data
ball_coords = load_points_from_file("ball_coords.txt")
bounce_data = load_bounce_data("bounces.txt")
print("LOADED BALL COORDS", len(ball_coords))
print("LOADED BOUNCE DATA", len(bounce_data))
start_inference(bounce_data, ball_coords,reference_points_test)
    