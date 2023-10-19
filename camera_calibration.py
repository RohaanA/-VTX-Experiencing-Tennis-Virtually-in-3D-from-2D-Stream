# Description: This file contains the functions used to calibrate the camera 
import numpy as np
import cv2 as cv

# this function takes in a list of points and returns a list of normalized points
# def normalize_points(points, width, height):
#     normalized_points = []
#     for point in points:
#         x = point[0] / width
#         y = point[1] / height
#         normalized_points.append((x, y))
#     return normalized_points

points = [
    (282, 854),
    (387, 663),
    (452, 853),
    (475, 505),
    (531, 662),
    (539, 390),
    (582, 313),
    (596, 504),
    (644, 389),
    (676, 312),
    (954, 312),
    (956, 389),
    (957, 504),
    (960, 662),
    (962, 853),
    (1236, 312),
    (1269, 389),
    (1319, 503),
    (1330, 312),
    (1374, 388),
    (1389, 661),
    (1440, 503),
    (1473, 853),
    (1533, 661),
    (1644, 853)
]

# Convert points to the required format
imgPts = np.array(points, dtype=np.float32)

print(imgPts)

unity_vertices = [
    (2.792, 0.2701, 2.153),
    (0.117, 0.2701, 2.153),
    (-7.999, 0.2701, 2.153),
    (-16.163, 0.2701, 2.153),
    (-18.889, 0.2701, 2.153),
    (2.792, 0.2701, -8.689),
    (0.117, 0.2701, -8.689),
    (-7.999, 0.2701, -8.689),
    (-16.163, 0.2701, -8.689),
    (-18.889, 0.2701, -8.689),
    (2.792, 0.2701, -21.392),
    (0.117, 0.2701, -21.392),
    (-7.999, 0.2701, -21.392),
    (-16.163, 0.2701, -21.392),
    (-18.889, 0.2701, -21.392),
    (2.792, 0.2701, -34.06),
    (0.117, 0.2701, -34.06),
    (-7.999, 0.2701, -34.06),
    (-16.163, 0.2701, -34.06),
    (-18.889, 0.2701, -34.06),
    (2.792, 0.2701, -44.881),
    (0.117, 0.2701, -44.881),
    (-7.999, 0.2701, -44.881),
    (-16.163, 0.2701, -44.881),
    (-18.889, 0.2701, -44.881)
]

# Convert vertices to the required format
objPts = np.array(unity_vertices, dtype=np.float32)
#1920x1080 
width = 1920
height = 1080
print(objPts)

def process():
	global C, fx, fd
	global C2
	global rvec, tvec
	global imgPts2

	C = cv.initCameraMatrix2D([objPts], [imgPts], (width, height))

	(rv, C2, distCoeffs, rvecs, tvecs) = cv.calibrateCamera(
		[objPts], [imgPts],
		imageSize=(1920, 1080),
		cameraMatrix=C.copy(),
		distCoeffs=np.float32([0,0,0,0,0]),
		flags=0
		| cv.CALIB_FIX_ASPECT_RATIO
		| cv.CALIB_FIX_PRINCIPAL_POINT
		| cv.CALIB_ZERO_TANGENT_DIST
		| cv.CALIB_FIX_K1
		| cv.CALIB_FIX_K2
        | cv.CALIB_USE_INTRINSIC_GUESS
		| cv.CALIB_FIX_K3
		| cv.CALIB_FIX_K4
		| cv.CALIB_FIX_K5
	)

	# C[0,0] = C[1,1] = C2[0,0]
	# C[1,2] = C2[1,2]
	# C[0,2] = C2[0,2]
	C = C2
	#print(np.linalg.norm(C-C2))

	print("distortion coefficients:")
	print(distCoeffs.T)
	print("camera matrix:")
	print(C)
	fx = C[0,0]

	# fx * tan(hfov/2) == width/2
	hfov = np.arctan(width/2 / fx) * 2
	print(f"horizontal FoV:\n\t{hfov / np.pi * 180:.2f} °")

	# x? mm focal length -> 36 mm horizontal (24 vertical)?
	fd = 36 / (np.tan(hfov/2) * 2)
	print(f"focal length (35mm equivalent):\n\t{fd:.2f} mm")

	(rv, rvec, tvec) = cv.solvePnP(objPts, imgPts, C, distCoeffs=None)
	print("tvec [m]:")
	print(tvec)

	(imgPts2, jac) = cv.projectPoints(
		objectPoints=objPts,
		rvec=rvec,
		tvec=tvec,
		cameraMatrix=C,
		distCoeffs=None)
process()
def process2():
    C = cv.initCameraMatrix2D([objPts], [imgPts], (width, height))
    print("camera matrix:")
    print(C)
    fx = C[0,0]

    # fx * tan(hfov/2) == width/2
    hfov = np.arctan(width/2 / fx) * 2
    print(f"horizontal FoV:\n\t{hfov / np.pi * 180:.2f} °")

    # x? mm focal length -> 36 mm horizontal (24 vertical)?
    fd = 36 / (np.tan(hfov/2) * 2)
    print(f"focal length (35mm equivalent):\n\t{fd:.2f} mm")

    (rv, rvec, tvec) = cv.solvePnP(objPts, imgPts, C, distCoeffs=None)
    print("tvec [m]:")
    print(tvec)
    
    return C, tvec, rvec, rv
old_C, old_tvec, old_rvec, old_rv = process2()

import numpy as np
import cv2

def map2Dto3D(points_2D, tvec, rvec, C, R):
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Invert the camera matrix
    C_inv = np.linalg.inv(C)

    # List to store 3D points
    points_3D = []

    # Iterate over each 2D point
    for point_2D in points_2D:
        # Convert 2D point to homogeneous coordinates
        point_2D_homo = np.array([point_2D[0], point_2D[1], 1])

        # Compute the intermediate 3D point
        intermediate_point = np.dot(np.dot(C_inv, R), point_2D_homo)

        # Compute the scale factor using the z-component of the translation vector
        scale_factor = tvec[2] / intermediate_point[2]

        # Compute the final 3D point
        point_3D = scale_factor * intermediate_point + tvec

        # Append the 3D point to the list
        points_3D.append(point_3D)

    # Write results to file
    with open("3dpoints.txt", "w") as file:
        for point_3D in points_3D:
            file.write(f"[{point_3D[0]}, {point_3D[1]}, {point_3D[2]}]\n")

    return points_3D
#Load all 2d poitns from "ball_coords.txt file"
import ast

#Loads files with entries like [973, 549], converts to 973, 549
#Ignores line having None
def load2DPointsFromFile(filename):
    points_2D = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if line != "None":
                point = line.strip("[]").split(",")
                x = int(point[0].strip())
                y = int(point[1].strip())
                points_2D.append((x, y))
    return points_2D
points2D = load2DPointsFromFile("ball_coords.txt")
print("LOADED BALL COORDS", points2D)
print(map2Dto3D(points2D, old_tvec, old_rvec, old_C, old_rv))