import socket
import time
import logging
import os

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Use environment variables for configuration or default to specified values
HOST = os.getenv('192.168.1.16', 'localhost')  # Default Raspberry Pi IP
PORT = int(os.getenv('SENSOR_PORT', '22'))  # Default port

def receive_sensor_data():
    """
    Connects to the server and continuously receives sensor data.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            logging.info(f"Connected to {HOST} on port {PORT}")

            while True:
                # Receive sensor data from Raspberry Pi
                data = s.recv(1024).decode()
                if not data:
                    logging.info("No data received. Connection may have been closed.")
                    break
                print(data.strip())  # Print the received sensor reading
                # Further processing and storing of the data can be done here
                time.sleep(1)  # Adjust interval for receiving updates

        except socket.error as e:
            logging.error(f"Socket error: {e}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
        finally:
            logging.info("Connection closed")

if __name__ == "__main__":
    receive_sensor_data()
