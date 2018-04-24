import sys
import socket
import torch as torch
from torch.multiprocessing import Process, Pipe
from ...utils import Timer
from ...utils import EventEmitter


host_ip = '10.102.20.26'
is_close = lambda pos, cue_pos: (pos[0]-cue_pos[0])**2 + (pos[1]-cue_pos[1]**2) < 8**2




class Jovian_Stream(str):
    def parse(self):
        line = self.__str__()
        _line = line.split(',')
        _t,_x,_y = int(_line[0]), int(_line[1]), int(_line[2])
        _coord = [_x, _y]
        return _t, _coord



class Jovian(EventEmitter):
    def __init__(self):
        super(Jovian, self).__init__()
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


    def _jovian_process(self):
        '''jovian reading process that use 
           a multiprocessing pipe + a jovian instance 
           as input parameters
        '''
        while True:
            with Timer('', verbose=True):
                self._t, self._coord = self.readline().parse()
                self.pipe_jovian_side.send((self._t, self._coord))
                self.examine_trigger()
            # print(self._t, self._coord)

    # def on_touch(self):
    #     print(self._t, self._coord)

    def examine_trigger(self):
        print(self._t, self._coord)
        if is_close(self._coord, (0,0)):
            self.emit('touch', args=(self._t, self._coord))


    def start(self):
        self.pipe_jovian_side, self.pipe_gui_side = Pipe()
        self.jovian_process = Process(target=self._jovian_process) #, args=(self.pipe_jovian_side,)
        self.jovian_process.daemon = True
        self.jovian_process.start()  


    def stop(self):
        self.jovian_process.terminate()
        self.jovian_process.join()
        self.reset()


    def get(self):
        return self.pipe_gui_side.recv()


    def teleport(self, prefix, target_pos, target_item=None):
        '''
           Core function: This is the only function that send `events` back to Jovian from interaction 
        '''
        x, y, z = target_pos # the coordination
        if prefix == 'console':  # teleport animal, target_item is not needed
            cmd = "{}.teleport({},{},{},{})\n".format(prefix, x,y,z,0)
            self.output.send(cmd)
            # print(cmd)
        elif prefix == 'model':  # move cue
            cmd = "{}.move('{}',{},{},{})\n".format(prefix, target_item, x, y, z)
            self.output.send(cmd)
            # print(cmd)
