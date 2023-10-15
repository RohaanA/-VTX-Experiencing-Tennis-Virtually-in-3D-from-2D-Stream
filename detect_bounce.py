import pandas as pd
import numpy as np
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from pickle import load
import cv2

def start_detectbounce(input_video_path, bounce_classifier_model, xy, V, output_path='outputs/bounce_classifier.mp4'):
  # Predicting Bounces 
  test_df = pd.DataFrame({'x': [coord[0] for coord in xy[:-1]], 'y':[coord[1] for coord in xy[:-1]], 'V': V})

  # df.shift
  for i in range(20, 0, -1): 
    test_df[f'lagX_{i}'] = test_df['x'].shift(i, fill_value=0)
  for i in range(20, 0, -1): 
    test_df[f'lagY_{i}'] = test_df['y'].shift(i, fill_value=0)
  for i in range(20, 0, -1): 
    test_df[f'lagV_{i}'] = test_df['V'].shift(i, fill_value=0)

  test_df.drop(['x', 'y', 'V'], 1, inplace=True)

  Xs = test_df[['lagX_20', 'lagX_19', 'lagX_18', 'lagX_17', 'lagX_16',
        'lagX_15', 'lagX_14', 'lagX_13', 'lagX_12', 'lagX_11', 'lagX_10',
        'lagX_9', 'lagX_8', 'lagX_7', 'lagX_6', 'lagX_5', 'lagX_4', 'lagX_3',
        'lagX_2', 'lagX_1']]
  Xs = from_2d_array_to_nested(Xs.to_numpy())

  Ys = test_df[['lagY_20', 'lagY_19', 'lagY_18', 'lagY_17',
        'lagY_16', 'lagY_15', 'lagY_14', 'lagY_13', 'lagY_12', 'lagY_11',
        'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6', 'lagY_5', 'lagY_4',
        'lagY_3', 'lagY_2', 'lagY_1']]
  Ys = from_2d_array_to_nested(Ys.to_numpy())

  Vs = test_df[['lagV_20', 'lagV_19', 'lagV_18',
        'lagV_17', 'lagV_16', 'lagV_15', 'lagV_14', 'lagV_13', 'lagV_12',
        'lagV_11', 'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6', 'lagV_5',
        'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1']]
  Vs = from_2d_array_to_nested(Vs.to_numpy())

  X = pd.concat([Xs, Ys, Vs], 1)

  # load the pre-trained classifier  
  clf = load(open(bounce_classifier_model, 'rb'))

  predcted = clf.predict(X)
  idx = list(np.where(predcted == 1)[0])
  idx = np.array(idx) - 10

  video = cv2.VideoCapture(input_video_path)

  output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(video.get(cv2.CAP_PROP_FPS))
  length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  fourcc = cv2.VideoWriter_fourcc(*'XVID')

  print(fps)
  print(length)

  output_video = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
  i = 0
  while True:
    ret, frame = video.read()
    if ret:
      # if coords[i] is not None:
      if i in idx:
        print("Drawing bounce....")
        center_coordinates = int(xy[i][0]), int(xy[i][1])
        radius = 3
        color = (255, 0, 0)
        thickness = -1
        cv2.circle(frame, center_coordinates, 10, color, thickness)
      i += 1
      output_video.write(frame)
    else:
      break

  video.release()
  output_video.release()
    