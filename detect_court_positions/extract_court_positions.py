import numpy as np
import cv2

# Points for the match stream image
stream_points = np.array([
    [582, 316], [1237, 316], [582, 857], [1237, 857]
], dtype=np.float32)

# Points for the reference court image
reference_points = np.array([
    [287, 562], [1241, 562], [287, 2935], [1241, 2935]
], dtype=np.float32)

# Convert the points to numpy arrays
stream_points = np.float32(stream_points)
reference_points = np.float32(reference_points)

# Calculate the transformation matrix
transform_matrix = cv2.getPerspectiveTransform(stream_points, reference_points)


#Set the court corner points for the stream image
def define_stream_points(point1, point2, point3, point4):
    stream_points = np.array([
        [point1], [point2], [point3], [point4]
    ], dtype=np.float32)
    return stream_points
#Set the court corner points for the reference image
def define_reference_points(point1, point2, point3, point4):
    reference_points = np.array([
        [point1], [point2], [point3], [point4]
    ], dtype=np.float32)
    return reference_points
# This function is responsible for mapping the given player coordinates from the stream image to the reference image
# Parameters: player1_coords, player2_coords
# playerX_coords: The bottom middle coordinate of the bounding box of player X
def extract_positions(player1_coords, player2_coords):
    if (player1_coords is not None) and (player2_coords is not None):
        # Apply the transformation matrix to the player coordinates
        mapped_player1_coords = np.dot(transform_matrix, player1_coords)
        mapped_player2_coords = np.dot(transform_matrix, player2_coords)
            
        # Normalize the mapped coordinates
        mapped_player1_coords /= mapped_player1_coords[2]
        mapped_player2_coords /= mapped_player2_coords[2]

        # Extract the x and y coordinates from the mapped coordinates
        mapped_player1_x, mapped_player1_y = mapped_player1_coords[0], mapped_player1_coords[1]
        mapped_player2_x, mapped_player2_y = mapped_player2_coords[0], mapped_player2_coords[1]
        
        # Return the mapped coordinates
        return mapped_player1_x, mapped_player1_y, mapped_player2_x, mapped_player2_y
    else:
        return None, None, None, None #Sign of error
