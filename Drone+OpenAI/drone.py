import socket
import time
import cv2
import threading
from openai import OpenAI

# UDP command setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
tello_address = ('192.168.10.1', 8889)
sock.bind(('', 9000))

# Send init commands
sock.sendto(b'command', tello_address)
time.sleep(1)
sock.sendto(b'streamon', tello_address)
time.sleep(1)

# Function to show video
def show_video():
    cap = cv2.VideoCapture("udp://@0.0.0.0:11111")

    if not cap.isOpened():
        print("Failed to open video stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received — retrying...")
            time.sleep(0.1)
            continue

        cv2.imshow("Tello Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start video thread as daemon
video_thread = threading.Thread(target=show_video, daemon=True)
video_thread.start()

# Control loop
try:
    print("Taking off...")
    sock.sendto(b'takeoff', tello_address)
    time.sleep(5)
    print("Landing...")
    sock.sendto(b'land', tello_address)

except KeyboardInterrupt:
    print("\nKeyboardInterrupt caught — landing and exiting...")
    sock.sendto(b'land', tello_address)

finally:
    sock.sendto(b'streamoff', tello_address)
    print("Stream off. Exiting now.")
