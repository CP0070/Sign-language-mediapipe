import socket

def client_program():
    # Create a socket object
    s = socket.socket()

    # Define the port on which you want to connect (match this with the server's port)
    port = 12345

    # Connect to the server on local computer
    server_ip = '192.168.1.9'  # Ensure this is the IP where the server is running
    s.connect((server_ip, port))

    try:
        while True:
            # Receive data from the server
            data = s.recv(2048)  # You might want to adjust the buffer size if needed
            if not data:
                print("No more data from server.")
                break  # Exit if no data is received, indicating server might have closed the connection

            # Decode and print the data
            print("Received sensor data:\n", data.decode('utf-8'))

    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        # Close the connection
        s.close()
        print("Connection closed.")

if __name__ == "__main__":
    client_program()
