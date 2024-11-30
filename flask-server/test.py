import os
import cv2
import pickle
import csv
import numpy as np
from datetime import datetime
from scipy.spatial import distance as dist
import dlib
import winsound  # For beep sound
import time

# Eye aspect ratio calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Blink detection function
def detect_blink(predictor, gray_frame, face):
    shape = predictor(gray_frame, face)
    shape = np.array([[p.x, p.y] for p in shape.parts()])
    left_eye = shape[36:42]
    right_eye = shape[42:48]
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

# Paths to data files
script_dir = os.path.dirname(os.path.abspath(__file__))
haarcascade_path = os.path.join(script_dir, 'data', 'haarcascade_frontalface_default.xml')
names_and_usns_path = os.path.join(script_dir, 'data', 'names_and_usns.pkl')
shape_predictor_path = os.path.join(script_dir, 'data', 'shape_predictor_68_face_landmarks.dat')
attendance_dir = os.path.join(script_dir, 'attendance')
os.makedirs(attendance_dir, exist_ok=True)

# Verify necessary files
if not all(os.path.exists(path) for path in [haarcascade_path, names_and_usns_path, shape_predictor_path]):
    print("Error: One or more necessary files are missing!")
    exit()

# Load names and USNs
with open(names_and_usns_path, 'rb') as f:
    names_and_usns = pickle.load(f)

# Initialize video capture and detector
video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(haarcascade_path)
predictor = dlib.shape_predictor(shape_predictor_path)
detector = dlib.get_frontal_face_detector()

# Load trained face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
try:
    recognizer.read('Trainer.yml')
except cv2.error as e:
    print(f"Error loading Trainer.yml: {e}")
    exit()

# Blink detection and liveness parameters
EAR_THRESHOLD = 0.2
CONSEC_FRAMES = 3
blink_counter = 0
liveness_confirmed = False
UNKNOWN_CONFIDENCE_THRESHOLD = 100  # Adjust for unknown face detection

# Recognition loop
attendance = set()  # To store recorded attendance for the session
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        face_image = gray[y:y + h, x:x + w]

        # Validate face_image
        if face_image.size == 0 or face_image.shape[0] <= 0 or face_image.shape[1] <= 0:
            print("Invalid face image detected. Skipping...")
            continue

        # Resize face_image to the size used during training
        face_image = cv2.resize(face_image, (100, 100))

        # Detect liveness through blink detection
        ear = detect_blink(predictor, gray, face)
        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= CONSEC_FRAMES:
                liveness_confirmed = True
            blink_counter = 0

        # Draw the rectangle around the face
        rectangle_color = (0, 0, 255)  # Red by default
        liveness_message = "Liveness not confirmed"
        if liveness_confirmed:
            rectangle_color = (0, 255, 0)  # Green if liveness confirmed
            liveness_message = "Liveness confirmed"

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)

        # Display liveness message on top-left corner
        cv2.putText(frame, liveness_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, rectangle_color, 2)

        # Perform face recognition if liveness confirmed
        if liveness_confirmed:
            print("Liveness confirmed. Starting 3-second countdown...")

            # Countdown timer without freezing the video feed
            countdown_start = time.time()
            while time.time() - countdown_start < 3:
                ret, frame = video_capture.read()
                if not ret:
                    break

                cv2.putText(frame, f"Attendance will be captured in: {3 - int(time.time() - countdown_start)}s",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Face Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    video_capture.release()
                    cv2.destroyAllWindows()
                    exit()

            # Predict the face
            label, confidence = recognizer.predict(face_image)
            print(f"Label: {label}, Confidence: {confidence}")

            if confidence > UNKNOWN_CONFIDENCE_THRESHOLD:
                # Display "Unknown" for unrecognized faces
                cv2.putText(frame, "Unknown Person", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("Unknown person detected!")
                cv2.imshow("Face Recognition", frame)
                cv2.waitKey(1000)
            else:
                # Recognized face processing
                if 0 <= label < len(names_and_usns):
                    full_name, usn = names_and_usns[label]
                    name, usn = full_name.split('_', 1)
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"Name: {name}\nUSN: {usn}\nTime: {current_time}")

                    # Display the name and USN on top of the face rectangle
                    cv2.putText(frame, f"{name} ({usn})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rectangle_color, 2)

                    if (name, usn) not in attendance:
                        attendance.add((name, usn, current_time))
                        winsound.Beep(1000, 500)
                        print(f"Attendance recorded for {name}")

                        # Save attendance to CSV
                        date = datetime.now().strftime('%Y-%m-%d')
                        file_path = os.path.join(attendance_dir, f"Attendance_{date}.csv")
                        file_exists = os.path.isfile(file_path)

                        with open(file_path, "a", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            if not file_exists:
                                writer.writerow(["Name", "USN", "Time"])
                            writer.writerows(attendance)

                        print("Attendance saved. Exiting...")
                        video_capture.release()
                        cv2.destroyAllWindows()
                        exit()
                else:
                    print(f"Error: Label {label} is out of range.")

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows() 