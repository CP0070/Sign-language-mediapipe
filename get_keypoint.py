import argparse
import cv2
import mediapipe as mp
import numpy as np
import time

from util import *

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument('--use_static_image_mode', action='store_true')
args = parser.parse_args()
use_static_image_mode = args.use_static_image_mode

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

prev_frame_time = 0
read = False

# For webcam input:
cap = cv2.VideoCapture(args.device)
with mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        pre_processed_landmark_list = None  # Reset for each frame

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                landmark_list = calc_landmark_list(image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_landmark_list = np.array(pre_processed_landmark_list)

        # Key press handling after processing the frame

        key = cv2.waitKey(10)
        if key == ord("t"):
            read = not read

        if read:
            cv2.putText(image, "Read: ON", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if (ord('0') <= key <= ord('9') or ord('A') <= key <= ord('Z')) and pre_processed_landmark_list is not None:
            number = key - 48 if key <= ord('9') else key - 55
            if read:

                    logging_csv(number, pre_processed_landmark_list)
                    cv2.putText(image, "Logging to CSV", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) != 0 else 0
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {fps:.2f}"

        image = cv2.flip(image, 1)
        cv2.putText(image, fps_text, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Hand Gesture Detection', image)

        if key & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
