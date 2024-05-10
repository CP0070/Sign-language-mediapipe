import streamlit as st
import subprocess

def main():
    st.title("Hand Gesture Recognition")

    if st.button("Predict"):
        st.write("Running prediction...")
        process = subprocess.Popen(["python", "main.py"], stdout=subprocess.PIPE)
        output, error = process.communicate()
        st.write(output.decode("utf-8"))
        if error:
            st.error(error.decode("utf-8"))


    if st.button("Gloves"):
        st.write("starting server")
        process = subprocess.Popen(["python", "model/sensor_classifier/Sensor_data_client.py"], stdout=subprocess.PIPE)
        output, error = process.communicate()
        st.write(output.decode("utf-8"))
        if error:
            st.error(error.decode("utf-8"))

if __name__ == "__main__":
    main()