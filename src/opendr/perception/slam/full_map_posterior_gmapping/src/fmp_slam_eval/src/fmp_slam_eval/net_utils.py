import socket

def next_free_port(host="localhost", port=1024, max_port=65535):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    p = port
    while p <= max_port:
        try:
            sock.bind((host, p))
            sock.close()
            return p
        except OSError:
            p += 1
    raise IOError('No available free ports found in the range {} - {}.'.format(port, max_port))