import os
import cv2
import numpy as np
from PIL import Image
import pickle

# Path setup
dataset_dir = "dataset"  # Directory containing subfolders with images for each person
trainer_path = "Trainer.yml"  # Path to save the trained model
names_and_usns_path = os.path.join("data", "names_and_usns.pkl")  # Save the label mapping

# Ensure the required directories exist
os.makedirs("data", exist_ok=True)

# Initialize the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_image_and_labels(dataset_dir):
    """
    Reads images and their labels from the dataset directory.
    Displays each image during processing to visualize the training phase.
    """
    image_paths = []
    labels = []
    label_map = {}  # Dictionary to map unique labels to individual folders
    current_label = 0  # Initialize label counter

    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue  # Skip non-folder files

        if folder not in label_map:  # Assign a new label for each unique person
            label_map[folder] = current_label
            current_label += 1

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if not image_name.endswith(('jpg', 'png', 'jpeg')):
                continue  # Skip non-image files

            # Add image path and corresponding label
            image_paths.append(image_path)
            labels.append(label_map[folder])

    faces = []  # List to store face images
    ids = []  # List to store corresponding labels

    for image_path, label in zip(image_paths, labels):
        # Read the image in grayscale
        face_image = Image.open(image_path).convert('L')
        face_np = np.array(face_image, 'uint8')
        faces.append(face_np)
        ids.append(label)

        # Display the current image being processed
        cv2.imshow("Training on image...", face_np)
        cv2.putText(face_np, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the window and wait briefly to give a visual cue for the training process
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to quit early
            break

    cv2.destroyAllWindows()
    return np.array(ids), faces, label_map

# Get the images and labels
ids, faces, label_map = get_image_and_labels(dataset_dir)

if len(faces) > 0:
    # Train the recognizer
    recognizer.train(faces, ids)
    recognizer.save(trainer_path)  # Save the trained model
    print(f"Training completed. Model saved at {trainer_path}.")

    # Save the label map to a file
    with open(names_and_usns_path, 'wb') as f:
        pickle.dump(list(label_map.items()), f)
    print(f"Label mapping saved to {names_and_usns_path}.")
else:
    print("No data found for training! Make sure the dataset directory is populated.")

# Summary
if len(label_map) > 0:
    print("Training Summary:")
    for folder, label in label_map.items():
        print(f"Label {label}: {folder}")
