# Description: This file contains the functions used to calibrate the camera 

# this function takes in a list of points and returns a list of normalized points
def normalize_points(points, width, height):
    normalized_points = []
    for point in points:
        x = point[0] / width
        y = point[1] / height
        normalized_points.append((x, y))
    return normalized_points

