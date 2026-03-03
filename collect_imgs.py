import os
import cv2

# Directory to store dataset
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

# Initialize video capture
cap = cv2.VideoCapture(1)  # Adjust index (0, 1, 2) if needed

if not cap.isOpened():
    print("Error: Unable to access the camera. Check camera connection or index.")
    exit()

for j in range(number_of_classes):
    # Create class-specific directories
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Wait for the user to get ready
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Capture dataset images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
