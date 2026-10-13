#!/usr/bin/env python3
# Unix-socket MITM proxy: record vocoder requests (n, codes[n,16]) to a log.
import os, socket, struct, sys, threading

LISTEN = sys.argv[1]      # e.g. /tmp/qwen3_voc.sock (what clients connect to)
UPSTREAM = sys.argv[2]    # e.g. /tmp/qwen3_voc_real.sock (real server)
LOG = sys.argv[3]         # request log

def recvn(s, n):
    b = b''
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            raise EOFError
        b += c
    return b

def pipe(src, dst):
    try:
        while True:
            dst.sendall(src.recv(65536))
    except Exception:
        try: dst.shutdown(socket.SHUT_WR)
        except Exception: pass

def handle(c):
    u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    u.connect(UPSTREAM)
    # request: [i32 n][int64 n*16]
    hdr = recvn(c, 4)
    n = struct.unpack('<i', hdr)[0]
    body = recvn(c, n * 16 * 8)
    with open(LOG, 'ab') as f:
        f.write(struct.pack('<i', n) + body)
    u.sendall(hdr + body)
    t = threading.Thread(target=pipe, args=(u, c)); t.start()
    pipe(c, u); t.join(); c.close(); u.close()

if os.path.exists(LISTEN):
    os.remove(LISTEN)
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.bind(LISTEN); s.listen(8)
while True:
    c, _ = s.accept()
    threading.Thread(target=handle, args=(c,), daemon=True).start()
