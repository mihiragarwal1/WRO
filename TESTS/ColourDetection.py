import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
   
    img = cv2.resize(img, (640, 480))
    # Make a copy to draw contour outline
    input_image_cpy = img.copy()
 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
    # define range of red color in HSV
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
     
    # define range of green color in HSV
    lower_green = np.array([40, 20, 50])
    upper_green = np.array([90, 255, 255])
      
    # create a mask for red color
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    # create a mask for green color
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # create a mask for blue color
 
    # find contours in the red mask
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contours in the green mask
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contours in the blue mask
 
    # loop through the red contours and draw a rectangle around them
    for cnt in contours_red:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, 'Red', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
 
    # loop through the green contours and draw a rectangle around them
    for cnt in contours_green:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, 'Green', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
 
    cv2.imshow('Color Recognition Output', img)
     
    # Close video window by pressing 'x'
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break