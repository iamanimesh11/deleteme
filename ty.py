import cv2
from deepface import DeepFace

# Load the Haarcascade classifier
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture from webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()  # Read one frame from the video stream
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Analyze emotions using DeepFace
        result = DeepFace.analyze(frame[y:y + h, x:x + w], actions=['emotion'])

        # Get the dominant emotion
        dominant_emotion = result[0]['dominant_emotion']

        # Display the dominant emotion text beside the face
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,
                    dominant_emotion,
                    (x + w + 10, y + h // 2),
                    font, 0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()