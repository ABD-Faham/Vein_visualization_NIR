import cv2
import numpy as np
import time


# Set Camera
camera = cv2.VideoCapture(0)
camera.set(3, 480)
camera.set(4, 320)

# Set up display window
cv2.namedWindow('Vein Visualization', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Vein Visualization', 480, 320)

time.sleep(2)

while True:

    ret, frame = camera.read()

# Background removal
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array([133,70,70])
    hsv_upper = np.array([255,230,230])
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)


# CLAHE Algorithm
    h, s,v =result[:,:,0], result[:,:,1], result[:,:,2]
    image_bw = cv2.createCLAHE(clipLimit=5, tileGridSize=(12,12)).apply(v)


# Smoothing - Bilateral filter
    bilateral = cv2.bilateralFilter(image_bw, 4, 200, 2)


# Adaptive Threshold
    thresh = cv2.adaptiveThreshold(bilateral, 
                              255, 
                              cv2.ADAPTIVE_THRESH_MEAN_C, 
                              cv2.THRESH_BINARY, 
                              27, 
                              6)


# Morphological Transformation
    kernel = np.ones((3,3), np.uint8)
    dilated_img = cv2.dilate(thresh, kernel, iterations=2)
    kernel = np.ones((2,2), np.uint8)
    morphed = cv2.morphologyEx(dilated_img, cv2.MORPH_OPEN, kernel)


# color
    color = cv2.cvtColor(morphed,cv2.COLOR_GRAY2RGB)
    color[np.all(color == (0, 0, 0), axis=-1)] = (0,100,250)


# Overlay
    masked = cv2.addWeighted(color, 0.2, result, 0.8, 2)

# Show Output
    cv2.imshow('Vein Visualization', masked)
    
    
    # Check for exit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()