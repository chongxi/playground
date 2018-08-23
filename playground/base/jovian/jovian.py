import sys
import socket
import numpy as np
import torch as torch
from torch.multiprocessing import Process, Pipe
from ...utils import Timer
from ...utils import EventEmitter

# from .task import *


ENABLE_PROFILER = False

host_ip = '10.102.20.26'
pynq_ip = '10.102.20.105'
verbose = True

is_close = lambda pos, cue_pos, radius: (pos-cue_pos).norm()/100 < radius


class Jovian_Stream(str):
    def parse(self):
        line = self.__str__()
        _line = line.split(',')
        try:
            _t,_x,_y = int(_line[0]), int(_line[1]), int(_line[2])
            _coord = [_x, _y, 0]
            return _t, _coord
        except:
            _t,_info = int(_line[0]), _line[1]
            return _t, _info


class Jovian(EventEmitter):
    '''
    Jovian is the abstraction of Remote Jovian software, it does following job:
    0.  jov = Jovian()                                      # instance
    1.  jov.readline().parse()                              # read from mouseover
    2.  jov.start(); jov.stop()                             # start reading process in an other CPU
    3.  jov.set_trigger(); jov.examine_trigger();           # set and examine the trigger condition (so far only touch) based on both current input and current state
    4.  jov.teleport(prefix, target_pos, target_item)       # execute output (so far only teleport)

    Jovian is a natrual event emit, it generate two `events`:
    1. touch      (according to input and trigger condition, it touches something)
    2. teleport   (based on the task fsm, something teleports)
    '''
    def __init__(self):
        super(Jovian, self).__init__()
        self.socket_init()
        self.buf_init()
        self.shared_mem_init()


    def socket_init(self):
        ### mouseover server connection
        self.input = socket.create_connection((host_ip, '22224'), timeout=1)
        self.input.setblocking(1)
        self.input.settimeout(0.8)
        self.input.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.output = socket.create_connection((host_ip, '22223'), timeout=1)
        self.output.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.output_control = socket.create_connection((host_ip, '22225'), timeout=1)
        self.enable_output()

        ### pynq server connection
        self.pynq = socket.create_connection((pynq_ip, '2222'), timeout=1)
        self.pynq.setblocking(1)
        self.pynq.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self.socks = [self.input,  self.output, self.output_control, self.pynq]


    def buf_init(self):
        self.buf = None        # the buf generator
        self.buffer = ''       # the content
        self.buffering = False # the buffering state


    def shared_mem_init(self):
        # trigger task using frame counter
        self.cnt = torch.empty(1,)
        self.cnt.share_memory_()
        # current position of animal
        self.current_pos = torch.empty(3,)
        self.current_pos.share_memory_()
        # the influence radius of the animal
        self.touch_radius = torch.empty(1,)
        self.touch_radius.share_memory_()


    def reset(self):
        [conn.shutdown(2) for conn in self.socks]
        self.buf_init()
        self.socket_init()


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
            with Timer('', verbose=ENABLE_PROFILER):
                self._t, self._coord = self.readline().parse()
                self.log.info('{}, {}'.format(self._t, self._coord))
                if type(self._coord) is list:
                    self.current_pos[:]  = torch.tensor(self._coord)
                    self.task_routine()
                # self.pipe_jovian_side.send((self._t, self._coord))


    def set_trigger(self, shared_cue_dict):
        '''shared_cue_dict is a a shared memory dict between processes contains cue name and position:
           shared_cue_dict := {cue_name: cue_pos, 
                               ...}
        '''
        self.shared_cue_dict = shared_cue_dict
        self.log.info('-----------------------------------------------------------------------------------------')
        self.log.info('jovian and maze_view is connected, they starts to share cues position and transformations')
        self.log.info('-----------------------------------------------------------------------------------------')


    def task_routine(self):
        self.cnt.add_(1)
        if self.cnt == 1:
            self.emit('start')
        # if self.cnt%2 == 0:
        self.emit('frame')
        self.examine_trigger()


    def examine_trigger(self):
        for _cue_name in self.shared_cue_dict.keys():
            # self.log.info('{},{}'.format(_cue_name, self.shared_cue_dict[_cue_name]))
            if self._is_close(self.current_pos, torch.tensor(self.shared_cue_dict[_cue_name]), self.touch_radius):
                # self.log.info('touch {}@{}'.format(_cue_name, self.shared_cue_dict[_cue_name]))
                self.emit('touch', args=(_cue_name, self.shared_cue_dict[_cue_name]))


    def start(self):
        self.pipe_jovian_side, self.pipe_gui_side = Pipe()
        self.jovian_process = Process(target=self._jovian_process, name='jovian') #, args=(self.pipe_jovian_side,)
        self.jovian_process.daemon = True
        self.reset() # !!! reset immediately before start solve the first time jam issue
        self.jovian_process.start()  


    def stop(self):
        self.jovian_process.terminate()
        self.jovian_process.join()
        self.cnt.fill_(0)


    def get(self):
        return self.pipe_gui_side.recv()


    def _is_close(self, pos, cue_pos, radius):
        return is_close(pos, cue_pos, radius)


    def toggle_motion(self):
        cmd = "console.toggle_motion()\n"
        self.output.send(cmd)


    def teleport(self, prefix, target_pos, target_item=None):
        '''
           Core function: This is the only function that send `events` back to Jovian from interaction 
        '''
        try:
            x, y, z = target_pos # the coordination
        except:
            x, y = target_pos
            z = 0

        if prefix == 'console':  # teleport animal, target_item is not needed
            cmd = "{}.teleport({},{},{},{})\n".format(prefix, x,y,5,0)
            self.output.send(cmd)

        elif prefix == 'model':  # move cue
            with Timer('', verbose = ENABLE_PROFILER):
                z += self.shared_cue_height[target_item]
                cmd = "{}.move('{}',{},{},{})\n".format(prefix, target_item, x, y, z)
                self.output.send(cmd)
                bottom = z - self.shared_cue_height[target_item]
                self.shared_cue_dict[target_item] = self._to_jovian_coord(np.array([x,y,bottom], dtype=np.float32))


    def reward(self, time):
        try:
            self.pynq.send('reward, {}'.format(time))
        except:
            pass
