import sys
import socket

import torch.multiprocessing as mp
from torch.multiprocessing import Process, Pipe

host_ip = '10.102.20.26'

class Jovian_Stream(str):
    def parse(self):
        line = self.__str__()
        _line = line.split(',')
        _t,_x,_y = int(_line[0]), int(_line[1]), int(_line[2])
        _coord = [_x, _y]
        return _t, _coord



class Jovian(object):
    def __init__(self):
        self.socket_init()
        self.buf_init()

    def socket_init(self):
        self.input = socket.create_connection((host_ip, '22224'), timeout=1)
        self.input.setblocking(1)
        self.input.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.output = socket.create_connection((host_ip, '22223'), timeout=1)
        self.output.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.output_control = socket.create_connection((host_ip, '22225'), timeout=1)
        self.enable_output()

    def buf_init(self):
        self.buf = None        # the buf generator
        self.buffer = ''       # the content
        self.buffering = False # the buffering state

    def reset(self):
        self.input.close()
        self.output.close()
        self.output_control.close()
        self.socket_init()
        self.buf_init()

    def enable_output(self, enable=True):
        if enable:
            self.output_control.send(b'1')
        else:
            self.output_control.send(b'1')

    def readbuffer(self):
        self.buffer = self.input.recv(256)
        self.buffering = True
        while self.buffering:
            if "\n" in self.buffer:
                (line, self.buffer) = self.buffer.split("\n", 1)
                yield Jovian_Stream(line + "\n")
            else:
                more = self.input.recv(256)
                if not more:
                    self.buffering = False
                else:
                    self.buffer += more
        if self.buffer:
            yield Jovian_Stream(self.buffer)


    def readline(self):
        if self.buf is None:
            self.buf = self.readbuffer()
            return self.buf.next()
        else:
            return self.buf.next()

    def get(self):
        return self.readline().parse()