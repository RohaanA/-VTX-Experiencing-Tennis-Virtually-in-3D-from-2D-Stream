import os

#This function extracts and saves the keypoints from the YOLOv8 model into a text file
def extract_keypoints(frameCount, results, output_directory):
    # Store keypoints as JSON in a text file
    keypoints_path = os.path.join(output_directory, f"frame{frameCount}.txt")
    with open(keypoints_path, 'w') as file:
        for r in results:
            keypoint_JSON = r.tojson(normalize=True)
            file.write(keypoint_JSON)
            file.write('\n')