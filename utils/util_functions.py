import re
from PIL import Image

def get_bounding_box_from_roi_file(file_path, image_shape):
    #get the image shape from the image
    image_shape = Image.open(image_shape).size

    # List to store all rectangles
    rectangles = []

    # Open the .txt file and read it line by line
    with open(file_path, 'r') as f:
        for line in f:
            # Extract the center, width, and height of the ROI
            match = re.search(r'center=\((.*), (.*)\), width=(.*), height=(.*)', line)
            center = (float(match.group(1)), float(match.group(2)))
            width = float(match.group(3))
            height = float(match.group(4))

            # Calculate the top-left and bottom-right corners of the rectangle
            top_left = (center[0] - width / 2, center[1] - height / 2)
            bottom_right = (center[0] + width / 2, center[1] + height / 2)

            # Add the rectangle to the list
            rectangles.append((top_left, bottom_right))

    # Calculate the minimum and maximum x and y coordinates
    min_x = min(rect[0][0] for rect in rectangles)
    min_y = min(rect[0][1] for rect in rectangles)
    max_x = max(rect[1][0] for rect in rectangles)
    max_y = max(rect[1][1] for rect in rectangles)

    # The minimum and maximum x and y coordinates are the top-left and bottom-right corners of the bounding box
    bounding_box = ((min_x, min_y), (max_x, max_y))

    # #add padding
    padding = 10
    bounding_box = ((bounding_box[0][0]-padding, bounding_box[0][1]-padding), (bounding_box[1][0]+padding, bounding_box[1][1]+padding))

    # # Check if the bounding box is within the image otherwise clip it
    bounding_box = (
        max(0, bounding_box[0][0]),
        max(0, bounding_box[0][1]),
        min(image_shape[0], bounding_box[1][0]),
        min(image_shape[1], bounding_box[1][1])
    )
    
    return bounding_box
    