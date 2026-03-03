import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: Model file './model.p' not found. Ensure the file exists.")
    exit()

# Initialize the camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Unable to access the camera. Check the camera index or ensure it is not in use.")
    exit()

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define label mapping
labels_dict = {0: 'hello', 1: 'iloveu', 2: 'yes'}

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Warning: Unable to read from the camera. Retrying...")
        continue

    # Frame dimensions
    H, W, _ = frame.shape

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Collect landmark data
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)

            # Normalize data based on min x, y
            min_x = min(x_)
            min_y = min(y_)

            for landmark in hand_landmarks.landmark:
                x = landmark.x - min_x
                y = landmark.y - min_y
                data_aux.append(x)
                data_aux.append(y)

        # Ensure data_aux matches the required input size
        expected_features = model.n_features_in_
        if len(data_aux) != expected_features:
            print(f"Error: Expected {expected_features} features, but got {len(data_aux)}.")
            continue

        # Bounding box
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Predict the character
        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue

        # Display prediction on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(
            frame,
            predicted_character,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

    # Show the frame
    cv2.imshow('frame', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
