import cv2

def start_detectvelocity(input_video_path, fps, output_width, output_height, V, output_path='outputs/velocity.mp4'):
    print("Detecting velocity...")
    # Load the output video
    velocity_video = cv2.VideoCapture(input_video_path)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    velocity_output_video = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    # Loop over frames from the velocity video
    while True:
        ret, frame = velocity_video.read()
        if ret is False:
            break

        # Get the frame number
        frame_number = int(velocity_video.get(cv2.CAP_PROP_POS_FRAMES))

        #Check if the frame number is valid
        if frame_number >= len(V):
            break
        # Get the corresponding velocity value
        velocity = V[frame_number]

        # Draw velocity information on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'Velocity: {velocity:.2f}'
        org = (10, 30)
        fontScale = 1
        color = (0, 255, 0)  # Green color
        thickness = 2

        # Draw a filled rectangle as the background
        rectangle_padding = 10
        text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
        rectangle_width = text_size[0] + 2 * rectangle_padding
        rectangle_height = text_size[1] + 2 * rectangle_padding
        rectangle_position = (org[0], org[1] - text_size[1] - rectangle_padding)
        cv2.rectangle(frame, rectangle_position, (rectangle_position[0] + rectangle_width, rectangle_position[1] + rectangle_height), (0, 0, 0), -1)

        # Write the velocity text on the frame
        cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

        # Write the frame to the output video
        velocity_output_video.write(frame)

    # Release the velocity video and output video
    velocity_video.release()
    velocity_output_video.release()
    