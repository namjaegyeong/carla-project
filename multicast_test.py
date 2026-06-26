import socket
import struct

GROUP = "239.255.0.1"
PORT = 7400

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("", PORT))

mreq = struct.pack("4s4s",
                   socket.inet_aton(GROUP),
                   socket.inet_aton("192.168.0.3"))

sock.setsockopt(socket.IPPROTO_IP,
                socket.IP_ADD_MEMBERSHIP,
                mreq)

print("Waiting...")

while True:
    data, addr = sock.recvfrom(4096)
    print(addr, data)