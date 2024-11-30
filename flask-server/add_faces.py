import os
import cv2
import dlib
import pickle
import numpy as np
from scipy.spatial import distance as dist

# Initialize video capture and face detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

# Constants
EAR_THRESHOLD = 0.2
CONSEC_FRAMES = 3
MAX_IMAGES = 100  # Number of images to collect per person
blink_counter = 0
liveness_confirmed = False
dataset_dir = "dataset"  # Directory to save images

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Prompt user for their name and USN
name = input("Enter Your Name: ")
usn = input("Enter Your USN: ")

# Validate dataset folder structure
person_dir = os.path.join(dataset_dir, f"{name}_{usn}")
os.makedirs(person_dir, exist_ok=True)

# Load existing data if available
if os.path.exists('data/names_and_usns.pkl'):
    with open('data/names_and_usns.pkl', 'rb') as f:
        names_and_usns = pickle.load(f)
else:
    names_and_usns = []

# Check if the (name, usn) pair already exists
if (name, usn) in names_and_usns:
    print("Error: This person is already registered with the same USN!")
    video.release()
    cv2.destroyAllWindows()
    exit()

# List to store face data
faces_data = []
image_count = 0

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Detect facial landmarks
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        # Get eye coordinates
        left_eye = shape[36:42]
        right_eye = shape[42:48]

        # Calculate EAR for both eyes
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)

        # Average EAR and check for blinks
        ear = (left_EAR + right_EAR) / 2.0
        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= CONSEC_FRAMES:
                liveness_confirmed = True
            blink_counter = 0

        # Draw bounding box and eyes on the frame
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        crop_img = frame[y:y + h, x:x + w, :]

        # Convert to grayscale for better recognition
        gray_crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_crop_img, (50, 50))

        # Save face data and image if liveness is confirmed
        if liveness_confirmed and image_count < MAX_IMAGES:
            faces_data.append(resized_img)
            image_path = os.path.join(person_dir, f"{name}_{usn}_{image_count + 1}.jpg")
            cv2.imwrite(image_path, gray_crop_img)
            print(f"Image saved: {image_path}")
            image_count += 1

        cv2.putText(frame, f"Liveness: {'Yes' if liveness_confirmed else 'No'}", 
                    (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Images Collected: {image_count}/{MAX_IMAGES}", 
                    (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or image_count == MAX_IMAGES:
        break

video.release()
cv2.destroyAllWindows()

# Save collected face data
faces_data = np.asarray(faces_data).reshape(len(faces_data), -1)
if len(faces_data) > 0:
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)

# Save the name and USN pair
names_and_usns.append((name, usn))
with open('data/names_and_usns.pkl', 'wb') as f:
    pickle.dump(names_and_usns, f)

print(f"Face registration for {name} ({usn}) completed and saved in {person_dir}.")
