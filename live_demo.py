import cv2
import dlib

# Initialize dlib's face detector and create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Get the current frame from the webcam
    _, frame = video_capture.read()

    # Use dlib's face detector to detect faces in the current frame
    faces = detector(frame)

    # Loop over the detected faces
    for face in faces:
        # Get the facial landmarks for the face
        landmarks = predictor(frame, face)

        # Crop the face using the facial landmarks
        face_cropped = cv2.resize(frame[landmarks.part(0).y:landmarks.part(16).y, landmarks.part(0).x:landmarks.part(16).x], (96, 96))

        # Crop the eyes using the facial landmarks
        eye1_cropped = cv2.resize(frame[landmarks.part(36).y:landmarks.part(41).y, landmarks.part(36).x:landmarks.part(39).x], (64, 96))
        eye2_cropped = cv2.resize(frame[landmarks.part(42).y:landmarks.part(47).y, landmarks.part(42).x:landmarks.part(45).x], (64, 96))

        # Display the cropped face and eyes
        cv2.imshow("Face", face_cropped)
        cv2.imshow("Eye 1", eye1_cropped)
        cv2.imshow("Eye 2", eye2_cropped)
        
        # TODO: Add logic to input this to the trained model and get the gaze point on the screen. 
        gaze_point = [-1, -1]
        
    # Draw a red dot at the gaze_point
    frame = cv2.circle(frame, gaze_point, 3, (0, 0, 255), -1)

    # Display the frame with the red dot
    cv2.imshow("Eye Tracking", frame)


    # Check if the user pressed "q" to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
