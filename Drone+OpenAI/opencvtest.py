import cv2
import socket
import time

# Start UDP socket to command Tello
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
tello_address = ('192.168.10.1', 8889)
sock.bind(('', 9000))
sock.sendto(b'command', tello_address)
time.sleep(1)
sock.sendto(b'streamon', tello_address)
time.sleep(1)

# Open video stream
cap = cv2.VideoCapture("udp://@0.0.0.0:11111")

# Initialize tracker
tracker = cv2.TrackerKCF_create()  # Try TrackerCSRT_create() for better accuracy

# Grab initial frame to select ROI
ret, frame = cap.read()
if not ret:
    print("Failed to grab frame from drone.")
    exit()

bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
tracker.init(frame, bbox)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    success, bbox = tracker.update(frame)
    if success:
        (x, y, w, h) = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Example logic: Move if box is too far from center
        frame_center_x = frame.shape[1] // 2
        object_center_x = x + w // 2
        offset = object_center_x - frame_center_x

        if abs(offset) > 50:  # tolerance
            if offset > 0:
                print("Move right")
                # sock.sendto(b'right 20', tello_address)
            else:
                print("Move left")
                # sock.sendto(b'left 20', tello_address)

    else:
        cv2.putText(frame, "Lost", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sock.sendto(b'streamoff', tello_address)
