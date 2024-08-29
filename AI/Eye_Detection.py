import cv2
import dlib

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    for face in faces:
        # Get the landmarks/parts for the face
        landmarks = predictor(gray, face)

        # Extract the coordinates of the left and right eyes
        left_eye = landmarks.part(36)
        right_eye = landmarks.part(45)

        # Draw circles around the eyes
        cv2.circle(frame, (left_eye.x, left_eye.y), 3, (0, 255, 0), -1)
        cv2.circle(frame, (right_eye.x, right_eye.y), 3, (0, 255, 0), -1)

    # Display the frame with detected eyes
    cv2.imshow("Eye Tracking", frame)

    # Break the loop if 'ESC' is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
