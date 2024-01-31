import cv2
import numpy as np

# Load the image
image = cv2.imread('/home/willows/Documents/study/semester3/advanced_image_processing/Modelling/react-project-python-backend/dataset/Dunkelbild/IO/Bild00003.BMP')

# List to store all ROIs
rois = []

# List of classes
classes = ['IO', 'NIO_Blase', 'NIO_Kerbe', 'NIO_Verbrennung']  # Modify this list as needed

# Infinite loop
while True:
    # Select a ROI
    r = cv2.selectROI(image)

    # If the ROI has zero width and height (i.e., the user has pressed ESC), break the loop
    if r[2] == 0 and r[3] == 0:
        break

    # Calculate the center pixel and save the center pixel and the width and height to the rois list
    center = (r[0] + r[2] / 2, r[1] + r[3] / 2)
    rois.append((center, r[2], r[3]))

# Save the classes and ROIs to a .txt file
with open('dataset/roi_dunkelbild.txt', 'w') as f:
    for i, roi in enumerate(rois):
        f.write('{}: center={}, width={}, height={}\n'.format(classes[i], roi[0], roi[1], roi[2]))

# Display the original image and the ROIs
cv2.imshow('image', image)
for i, roi in enumerate(rois, 1):
    cv2.imshow('roi_{}'.format(i), roi)

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()