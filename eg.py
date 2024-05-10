import socket
import csv


def client_program(server_ip, port, output_file):
    # Create a socket object with context manager for automatic cleanup
    with socket.socket() as s:
        # Connect to the server on local computer
        s.connect((server_ip, port))

        # Open a CSV file to store the data
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)

            try:
                while True:
                    # Receive data from the server
                    data = s.recv(2048)  # Adjust buffer size if needed
                    if not data:
                        print("No more data from server.")
                        break  # Exit if no data is received

                    # Decode data
                    decoded_data = data.decode('utf-8')
                    print("Received sensor data:\n", decoded_data)

                    # Write data to CSV file
                    writer.writerow([decoded_data])

            except Exception as e:
                print("An error occurred:", str(e))
            finally:
                print("Connection closed.")


if __name__ == "__main__":
    server_ip = '192.168.1.9'  # Ensure this is the IP where the server is running
    port = 12345
    output_file = 'sensor_data.csv'  # Specify the path to your output file
    client_program(server_ip, port, output_file)
