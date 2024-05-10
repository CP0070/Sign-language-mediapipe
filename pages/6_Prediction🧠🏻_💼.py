import streamlit as st
import subprocess

def main():
    st.title("👋 Hand Gesture Recognition 👋")

    # Add a colorful and engaging description
    st.markdown(
        """
        Welcome to Hand Gesture Recognition ! 🎉🤩

        This app allows you to predict hand gestures using trained models. Fun fact: Did you know that hand gestures
        can convey emotions, commands, and even cultural meanings?

        Choose an option below:

        - **Predict:** Click to predict your hand gesture.
        - **Gloves:** Experience real-time gesture recognition with sensor-equipped gloves. 🧤✨
        """
    )

    # Define functions for Predict and Gloves options
    if st.button("🔮 Predict"):
        predict_gestures()
    elif st.button("🧤 Gloves"):
        monitor_sensor_data()

def predict_gestures():
    st.write("Running prediction...")
    process = subprocess.Popen(["python", "main.py"], stdout=subprocess.PIPE)
    output, error = process.communicate()
    st.write(output.decode("utf-8"))
    if error:
        st.error(error.decode("utf-8"))

def monitor_sensor_data():
    st.write("Starting gloves for real-time gesture recognition...")
    process = subprocess.Popen(["python", "model/sensor_classifier/Sensor_data_client.py"], stdout=subprocess.PIPE)
    output, error = process.communicate()
    st.write(output.decode("utf-8"))
    if error:
        st.error(error.decode("utf-8"))

if __name__ == "__main__":
    main()
