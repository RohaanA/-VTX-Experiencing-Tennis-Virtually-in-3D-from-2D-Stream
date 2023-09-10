#Converts CVAT to COCO keypoints
#Make sure no singlular box tag is present

import xml.etree.ElementTree as ET
import os


# def convert_to_yolo(image_element, image_width, image_height, class_dict):
#     yolo_annotations = []

#     points = image_element.findall('points')
#     boxes = image_element.findall('box')

#     for i in range(len(points)):
#         points_elem = points[i]
#         box_elem = boxes[i] if i < len(boxes) else None

#         label = points_elem.get('label')
#         class_index = class_dict[label]

#         xtl = float(box_elem.get('xtl'))
#         ytl = float(box_elem.get('ytl'))
#         xbr = float(box_elem.get('xbr'))
#         ybr = float(box_elem.get('ybr'))

#         y_center = (ytl + ybr) / 2
#         w = xbr - xtl
#         h = ybr - ytl
#         x_center = (xtl + (w / 2)) / image_width
#         y_center = (ytl + (h / 2)) / image_height

#         w /= image_width
#         h /= image_height

#         # Normalize the points
#         points_str = points_elem.get('points')
#         normalized_points = []
#         for point in points_str.split(';'):
#             px, py = point.split(',')
#             px_normalized = float(px) / image_width
#             py_normalized = float(py) / image_height
#             normalized_points.append(f"{px_normalized},{py_normalized}")
#         normalized_points_str = ';'.join(normalized_points)

#         yolo_annotation = f"{class_index} {x_center} {y_center} {w} {h} {normalized_points_str}"
#         yolo_annotations.append(yolo_annotation)

#     return yolo_annotations

def convert_to_yolo(image_element, image_width, image_height, class_dict):
    
    yolo_annotations = []
    boxes = image_element.findall('box')
    skeletons= image_element.findall('skeleton')

    i=0

    for skeleton in skeletons:

        points = skeleton.findall('points')

        width = int(image_element.get('width'))
        height = int(image_element.get('height'))
        x_values = [float(point.get('points').split(',')[0]) for point in points]
        y_values = [float(point.get('points').split(',')[1]) for point in points]
        occlusion_values = [2 - int(point.get('occluded')) for point in points] 

        box_elem = boxes[i] if i < len(boxes) else None

        label = skeleton.get('label')
        class_index = class_dict[label]

        xtl = float(box_elem.get('xtl'))
        ytl = float(box_elem.get('ytl'))
        xbr = float(box_elem.get('xbr'))
        ybr = float(box_elem.get('ybr'))
        w = xbr - xtl
        h = ybr - ytl
        # x_center = (xtl + (w / 2)) / image_width
        # y_center = (ytl + (h / 2)) / image_height

        # Normalize the points

        normalized_points = []
        for index in range(len(x_values)):
            px=x_values[index]
            py=y_values[index]
            px_normalized = float(px) / width
            py_normalized = float(py) / height
            normalized_points.append(f"{px_normalized} {py_normalized} {(occlusion_values[index])}")
        normalized_points_str = ' '.join(normalized_points)

        yolo_annotation = f"{class_index} {(str((xtl + (w / 2)) / width))} {str((ytl + (h / 2)) / height)} {str(w / width)} {str(h / height)} {normalized_points_str}"
        yolo_annotations.append(yolo_annotation)

        i=i+1

    return yolo_annotations

# Path to the directory containing XML annotation files
annotations_dir = './'

# Image dimensions
image_width = 1920
image_height = 1080

# Define the class labels and their corresponding numerical class indices
class_dict = {
    'Player 1': 0,
    'Player 2': 0,
    # Add more labels as needed
}

# Iterate over XML files in the directory
for filename in os.listdir(annotations_dir):
    if filename.endswith('.xml'):
        xml_file = os.path.join(annotations_dir, filename)

        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Iterate over image elements in the XML
        for image_element in root.findall('image'):
            image_name = image_element.get('name')
            yolo_annotations = convert_to_yolo(image_element, image_width, image_height, class_dict)

            # Create a text file with YOLO annotations
            yolo_file = os.path.join("./test/labels", 'v2' + os.path.splitext(image_name)[0] + '.txt')
            with open(yolo_file, 'w') as f:
                for annotation in yolo_annotations:  
                    annotation = annotation.replace(';', ' ').replace(',', ' ')  # Remove semicolons and commas
                    f.write(annotation + '\n')
            

                               