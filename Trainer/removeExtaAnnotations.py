#First USE THIS ON THE ANNOTATIONS.XML file to remove un annotated frames

import xml.etree.ElementTree as ET

def remove_empty_images(annotations_xml_path):
    # Load the annotations XML file
    tree = ET.parse(annotations_xml_path)
    root = tree.getroot()

    # Find and remove image tags without any box tags
    removed_count = 0
    for image in root.findall('image'):
        if len(image.findall('skeleton')) == 0:
            root.remove(image)
            removed_count += 1

    # Save the modified annotations XML file
    tree.write(annotations_xml_path)

    print(f"Removal complete. {removed_count} image tags without any box tags were removed.")


# Specify the path to the annotations XML file
annotations_xml_path = "./annotations.xml"

# Call the function to remove empty image tags
remove_empty_images(annotations_xml_path)