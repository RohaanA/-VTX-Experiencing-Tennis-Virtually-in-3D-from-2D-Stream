import cv2
import os


def extract_frames(video_path, frames_file, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the video
    video = cv2.VideoCapture(video_path)

    # Check if the video is successfully opened
    if not video.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Read the frames file
    with open(frames_file, 'r') as f:
        frames_list = [int(os.path.splitext(frame.strip())[0].split('_')[-1]) + 1 for frame in f.readlines()]

    # Extract the selected frames
    frame_counter = 0
    frame_id = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        if frame_counter == frames_list[frame_id]:
            # Save the extracted frame as an image
            frame_output_path = os.path.join(output_dir, f"v2frame_{frame_counter - 1:06d}.jpg")
            cv2.imwrite(frame_output_path, frame)

            # Move to the next frame in the list
            frame_id += 1

            # Break the loop if all frames have been extracted
            if frame_id >= len(frames_list):
                break

        frame_counter += 1

    # Release the video capture
    video.release()


# Path to the video file
video_path = './v2.mp4'

# Path to the text file containing the list of frames to extract
frames_file = './sorted_file.txt'

# Directory to save the extracted frames
output_dir = './data/images'

# Extract the frames
extract_frames(video_path, frames_file, output_dir)