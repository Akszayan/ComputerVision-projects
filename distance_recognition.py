# ============================================
# Face Distance Estimation Using FaceMesh
# Author: Akszayan
# Description: This script uses the FaceMesh module from cvzone to detect facial landmarks
#              and estimate the distance from the camera based on the width between two points.
# Requirements: OpenCV, cvzone, mediapipe
# ============================================

import cv2
from cvzone.FaceMeshModule import FaceMeshDetector  # pip install cvzone

# Initialize video capture and FaceMesh detector
cap = cv2.VideoCapture(0)  # Start webcam
detector = FaceMeshDetector(maxFaces=1)  # Detect a maximum of one face

# Define the actual width of the face in cm (distance estimation)
W = 6.3  # Average distance between points in cm
f = 840  # Focal length (calibrated manually)

while True:
    success, img = cap.read()  # Read frame from webcam

    if not success:
        print("Failed to capture video.")
        break

    # Detect face mesh
    img, faces = detector.findFaceMesh(img)

    if faces:
        face = faces[0]  # Get the first detected face
        pointLeft = face[145]  # Left eye inner corner
        pointRight = face[374]  # Right eye inner corner
        nose = face[1]  # Nose tip
        
        # Draw landmarks on the image
        cv2.circle(img, nose, 5, (255, 0, 255), cv2.FILLED)

        # Calculate the distance between the two points
        w, _ = detector.findDistance(pointLeft, pointRight)  # Width in pixels

        # Estimate the distance to the face
        d = (W * f) / w
        distance = int(d - 25)  # Adjusted for better accuracy
        cv2.putText(img, f"Distance: {distance} cm", 
                    (face[10][0] - 75, face[10][1] - 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

    # Display the image
    cv2.imshow('Face Distance Estimation', img)

    # Exit the loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
