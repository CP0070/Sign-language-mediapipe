import socket
import asyncio
import json

async def receive_sensor_data():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(('0.0.0.0', 2222))
        print("Client is up and listening...")
        while True:
            data, _ = sock.recvfrom(4096)
            sensor_data = json.loads(data.decode('utf-8'))
            print("Received data:")
            for key, value in sensor_data.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for subkey, subvalue in value.items():
                        print(f"  {subkey}: {subvalue}")
                else:
                    print(f"{key}: {value}")
            await asyncio.sleep(0)  # Yield control to allow other tasks

def main():
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(receive_sensor_data())
    except KeyboardInterrupt:
        print("Client shutting down...")
    finally:
        loop.close()

if __name__ == "__main__":
    main()
