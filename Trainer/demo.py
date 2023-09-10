from ultralytics import YOLO
import cv2
import torch
if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")


# torch.cuda.set_device(0)    
# # # Load a model
#model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)

model.train(data="./config.yaml",epochs=300, imgsz=1280)


# # Open the video file
# video_path = "./v2.mp4"
# cap = cv2.VideoCapture(video_path)

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 inference on the frame
#         results = model(frame)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         #cv2.imshow("YOLOv8 Inference", annotated_frame)
#         dimension = (1280, 720)
#         resizedImage=cv2.resize(annotated_frame,dimension)
#         cv2.imshow("YOLOv8 Inference", resizedImage)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()