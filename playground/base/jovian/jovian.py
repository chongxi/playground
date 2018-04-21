import sys
import socket

import torch.multiprocessing as mp
from torch.multiprocessing import Process, Pipe

host_ip = '10.102.20.26'

class Jovian(object):
    def __init__(self):
        self.input = socket.create_connection((host_ip, '22224'), timeout=1)
        self.input.setblocking(1)
        self.input.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.output = socket.create_connection((host_ip, '22223'), timeout=1)
        self.output.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.output_control = socket.create_connection((host_ip, '22225'), timeout=1)

    def enable_output(self, enable=True):
        if enable:
            self.output_enable.send(b'1')
        else:
            self.output_enable.send(b'1')

    def readbuffer(self):
        buffer = self.input.recv(256)
        buffering = True
        while buffering:
            if "\n" in buffer:
                (line, buffer) = buffer.split("\n", 1)
                yield line + "\n"
            else:
                more = self.input.recv(256)
                if not more:
                    buffering = False
                else:
                    buffer += more
        if buffer:
            yield buffer

    def readline(self):
        for line in self.readbuffer():
            return line

    def get(self):
        for line in self.readbuffer():
            _line = line.split(',')
            _t,_x,_y = int(_line[0]), int(_line[1]), int(_line[2])
            _coord = [_x, _y]
            return _t, _coord