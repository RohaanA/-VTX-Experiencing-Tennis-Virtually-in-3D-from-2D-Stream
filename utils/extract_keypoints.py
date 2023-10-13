import os

#Generate a function called extract_keypoints
def extract_keypoints(results, output_directory):
    frameCount = -1
    # Store keypoints as JSON in a text file
    keypoints_path = os.path.join(output_directory, f"frame{frameCount}.txt")
    with open(keypoints_path, 'w') as file:
        for r in results:
            keypoint_JSON = r.tojson(normalize=True)
            file.write(keypoint_JSON)
            file.write('\n')