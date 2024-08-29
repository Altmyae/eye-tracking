import cv2
import numpy as np

cap = cv2.VideoCapture("eye_recording.flv")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Define the region of interest (ROI)
    roi = frame[269:795, 537:1416]
    rows, cols, _ = roi.shape

    # Convert to grayscale and apply Gaussian blur
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    # Apply binary thresholding
    _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        # Draw rectangle around the detected contour
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw crosshair at the center of the detected object
        cv2.line(roi, (x + w // 2, 0), (x + w // 2, rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + h // 2), (cols, y + h // 2), (0, 255, 0), 2)

        # Only process the first detected contour (largest one)
        break

    # Display the results
    cv2.imshow("Threshold", threshold)
    cv2.imshow("Gray ROI", gray_roi)
    cv2.imshow("ROI", roi)

    # Exit loop if 'Esc' is pressed
    key = cv2.waitKey(30)
    if key == 27:  # ASCII for 'Esc' key
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
