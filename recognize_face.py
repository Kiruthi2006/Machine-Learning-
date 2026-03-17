import cv2
import os
from collections import Counter

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Load trained face recognition model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("face_model.yml")

# Load names from dataset (sorted to match training labels)
names = sorted(os.listdir("dataset"))

# Start webcam
cam = cv2.VideoCapture(0)

# Store last predictions for stability
predictions = []

while True:

    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # Crop and resize face
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        # Predict face
        label, confidence = model.predict(face)

        print("Confidence:", confidence)

        # Decide name based on confidence
        if confidence < 120:
            name = names[label]
        else:
            name = "Unknown"

        # Add prediction to list
        predictions.append(name)

        # Keep only last 10 predictions
        if len(predictions) > 10:
            predictions.pop(0)

        # Majority vote for stability
        final_name = Counter(predictions).most_common(1)[0][0]

        # Display text with confidence
        text = f"{final_name} ({round(confidence,2)})"

        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    # Show camera window
    cv2.imshow("Face Recognition", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

# Release camera
cam.release()
cv2.destroyAllWindows()