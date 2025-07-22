import socket

RESP = {
    b'command':  b'ok',
    b'streamon': b'ok',
    b'streamoff':b'ok',
    b'takeoff':  b'ok',
    b'land':     b'ok',
}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 8889))
print("Fake Tello listening on 127.0.0.1:8889 … Ctrl+C to quit.")
while True:
    data, addr = sock.recvfrom(1024)
    print(f"[FAKE RX] {data!r} from {addr}")
    resp = RESP.get(data.strip(), b'ok')  # default ok
    sock.sendto(resp, addr)
