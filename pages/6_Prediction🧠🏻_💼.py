import subprocess
import csv
import cv2
import streamlit as st
import mediapipe as mp
import time
import numpy as np
from util import *
from model.keypoint_classifier.keypoint_classifier import *
from model.point_history_classifier.pont_history_classifier import *
from collections import deque
from utils import CvFpsCalc
from collections import Counter
from main import draw_landmarks, draw_info_text ,draw_info,draw_point_history,update_gestures
from Sensor_data_client import *

st.markdown("""
        <style>img {
        border-radius: 1rem;
        }</style>""",unsafe_allow_html=True)

def main():
    st.title("👋 Hand Gesture Recognition 👋")

    # Add a colorful and engaging description
    st.markdown(
        """
        Welcome to Hand Gesture Recognition! 🎉🤩

        This app allows you to predict hand gestures using trained models. Fun fact: Did you know that hand gestures
        can convey emotions, commands, and even cultural meanings?

        Let's get started! 🚀
        """
    )

    # Create columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        predict = st.button("🔮 Predict")
        # gloves_on = st.toggle("🧤 Gloves on")

    with col2:
        # Display an image showing examples of hand gestures
        st.image("https://github.com/githubhosting/ds/assets/126514044/125a0609-2cc6-49aa-9578-b8fe6cdca321",
                 caption="Examples of Hand Gestures", width=400)

    if predict:
        demo()

    st.title("Socket Client Interface")

    # Display input fields for server IP and port
    server_ip = st.text_input("Server IP", "172.20.10.2")
    port = st.number_input("Port", min_value=1024, max_value=65535, value=2222)

    # Display buttons for connect and disconnect actions
    connect_button = st.button("Connect")
    disconnect_button = st.button("Disconnect")

    if connect_button:
        # Here you would initiate the connection
        st.success(f"Attempting to connect to {server_ip} on port {port}...")
        # Placeholder for successful connection message
        st.success("Connected successfully.")

    if disconnect_button:
        # Placeholder for disconnection logic
        st.warning("Disconnected successfully.")
    #
    # if gloves_on:
    #     monitor_sensor_data()
def predict_gestures():
    st.write("Running prediction...")
    process = subprocess.Popen(["python", "main.py"], stdout=subprocess.PIPE)
    output, error = process.communicate()
    st.write(output.decode("utf-8"))
    if error:
        st.error(error.decode("utf-8"))

def demo():
    st.write("Running prediction...")
    stframe = st.empty()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    cap = cv2.VideoCapture(0)  # Replace 0 with the appropriate camera index
    if not cap.isOpened():
        st.error("Could not open the camera.")
        return
    accumulated_gestures = deque()  # List to store gesture texts
    last_gesture_time = time.time()
    gesture_cooldown = 2.0  # 1 second cooldown between gestures

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
            fps = cvFpsCalc.get()
            success, image = cap.read()
            if not success:
                st.warning("Could not read the camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            display_text = ""
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

                    # print(info_text)
                    # accumulation
                    gesture_detected = info_text
                    accumulated_gestures, last_gesture_time, display_text = update_gestures(
                        accumulated_gestures, last_gesture_time, gesture_detected, gesture_cooldown)


            else:
                point_history.append([0, 0])

            #new_frame_time = time.time()

            image = draw_point_history(image, point_history)
            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)



            # Display the accumulated gestures
            #display_text = ' '.join(filter(None, accumulated_gestures)).strip()
            #text_position = (10, image.shape[0] - 10)
            cv2.putText(image, display_text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            cv2.putText(image, "FPS:" + str(fps), (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3,
                                cv2.LINE_AA)
            #cv2.putText(image, display_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            stframe.image(image, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()
# def monitor_sensor_data():
#     st.write("Starting gloves for real-time gesture recognition...")
#     process = subprocess.Popen(["python", "model/sensor_classifier/Sensor_data_client.py"], stdout=subprocess.PIPE)
#     output, error = process.communicate()
#     st.write(output.decode("utf-8"))
#     if error:
#         st.error(error.decode("utf-8"))

if __name__ == "__main__":
    main()
    client_program()
