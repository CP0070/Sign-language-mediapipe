import csv
import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
from util import *
from model.keypoint_classifier.keypoint_classifier import *
from model.point_history_classifier.pont_history_classifier import *
from collections import deque
from utils import CvFpsCalc
from collections import Counter
from main import draw_landmarks, draw_info_text ,draw_info

# def draw_info(image, landmarks, info_text):
#
#     return image

def main():
    st.set_page_config(page_title="Hand Gesture Recognition")
    st.title("Hand Gesture Recognition")
    stframe = st.empty()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)  # Replace 0 with the appropriate camera index
    if not cap.isOpened():
        st.error("Could not open the camera.")
        return

    # Initialize classifiers and label lists
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]

    # Coordinate history
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.6
    ) as hands:
        while True:
            success, image = cap.read()
            if not success:
                st.warning("Could not read the camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Gesture recognition and processing
                    landmark_list = calc_landmark_list(image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(image, point_history)

                    confidence, hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == 1:
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    if confidence > 0.75:
                        info_text = keypoint_classifier_labels[hand_sign_id - 1]
                    else:
                        info_text = None

                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(finger_gesture_history).most_common()

                    image = draw_info(image, hand_landmarks, info_text)
                    image = draw_landmarks(image, landmark_list)
                    image = draw_info_text(image, point_history_classifier_labels[most_common_fg_id[0][0]])

                    stframe.image(image, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()