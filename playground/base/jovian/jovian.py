import sys
import socket
import numpy as np
import torch as torch
from torch.multiprocessing import Process, Pipe
from spiketag.utils import Timer
from spiketag.utils import EventEmitter
from spiketag.analysis.core import get_hd

ENABLE_PROFILER = False

host_ip = '10.102.20.42'
pynq_ip = '10.102.20.75'
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
        # self.input.setblocking(False)
        # self.input.settimeout(0.8)
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
        self.cnt.fill_(0)
        # current position of animal
        self.current_pos = torch.empty(3,)
        self.current_pos.share_memory_()

        # the influence radius of the animal
        self.touch_radius = torch.empty(1,)
        self.touch_radius.share_memory_()

        # bmi position (decoded position of the animal)
        self.bmi_pos = torch.empty(2,)
        self.bmi_pos.share_memory_()
        self.bmi_pos.fill_(0) 

        # bmi head-direction (inferred head direction at bmi_pos)
        self.hd_window = torch.empty(1,)  # time window(seconds) used to calculate head direction
        self.hd_window.share_memory_()
        self.hd_window.fill_(1)
        self.bmi_hd = torch.empty(1,)       # calculated hd sent to Jovian for VR rendering
        self.bmi_hd.share_memory_()
        self.bmi_hd.fill_(0)         
        self.current_hd = torch.empty(1,)   # calculated hd (same as bmi_hd) sent to Mazeview for local playground rendering
        self.current_hd.share_memory_()

        # bmi radius (largest teleportation range)
        self.bmi_teleport_radius = torch.empty(1,)
        self.bmi_teleport_radius.share_memory_()
        self.bmi_teleport_radius.fill_(0)    


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
        self.buffer = self.input.recv(256).decode("utf-8")
        self.buffering = True
        while self.buffering:
            if '\n' in self.buffer:
                (line, self.buffer) = self.buffer.split("\n", 1)
                yield Jovian_Stream(line + "\n")
            else:
                more = self.input.recv(256).decode("utf-8")
                if not more:
                    self.buffering = False
                else:
                    self.buffer += more
        if self.buffer:
            yield Jovian_Stream(self.buffer)


    def readline(self):
        if self.buf is None:
            self.buf = self.readbuffer()
            return self.buf.__next__()
        else:
            return self.buf.__next__()


    def _jovian_process(self):
        '''jovian reading process that use 
           a multiprocessing pipe + a jovian instance 
           as input parameters
        '''
        while True:
            with Timer('', verbose=ENABLE_PROFILER):
                try:
                    self._t, self._coord = self.readline().parse()
                except:
                    self.log.info('socket time out')
                if type(self._coord) is list:
                    self.log.info('{}, {}'.format(self._t, self._coord))
                else:
                    self.log.warn('{}, {}'.format(self._t, self._coord))
                if type(self._coord) is list:
                    self.current_pos[:]  = torch.tensor(self._coord)
                    self.task_routine()


    def set_bmi(self, bmi, pos_buffer_len=80):
        '''
        This set BMI, Its binner and decoder event for JOV to act on. The event flow:
        bmi.binner.emit('decode', X) ==> jov
        jov.emit('bmi_update', y)    ==> task (e.g. JEDI, JUMPER etc.)

        set_bmi connect the event flow from
                decode(X)               bmi_update(y)
        bmi ==================> jov ====================> task
        '''
        self.bmi = bmi
        self.bmi_pos_buf = np.zeros((pos_buffer_len, 2))
        hd_buffer_len = int(self.hd_window[0]/self.bmi.binner.bin_size)
        self.bmi_hd_buf  = np.zeros((hd_buffer_len, 2))
        self.log.info('initiate the BMI decoder and playground jov connection')
        @self.bmi.binner.connect
        def on_decode(X):
            '''
            This event is triggered every time a new bin is filled (based on BMI output timestamp)
            '''
            # print(self.binner.nbins, self.binner.count_vec.shape, X.shape, np.sum(X))
            with Timer('decoding', verbose=False):
                if self.bmi.dec.name == 'NaiveBayes':
                    X = np.sum(X, axis=0)
                # decode predict at current bin
                y = self.bmi.dec.predict_rt(X)
                #################### just for dusty test #########################
                y -= np.array([318.5,195.7])
                y /= 3
                ##################################################################
                # decide the output 
                self.bmi_pos_buf = np.vstack((self.bmi_pos_buf[1:, :], y))
                _teleport_pos = np.mean(self.bmi_pos_buf, axis=0)
                # set shared variable
                self.bmi_pos[:] = torch.tensor(_teleport_pos)
                self.bmi_hd_buf = np.vstack((self.bmi_hd_buf[1:, :], _teleport_pos))
                hd, speed = get_hd(trajectory=self.bmi_hd_buf, speed_threshold=0.6, offset_hd=180)
                if speed > .6:
                    self.bmi_hd[:] = torch.tensor(hd)      # sent to Jovian
                    self.current_hd[:] = torch.tensor(hd)  # sent to Mazeview
                # self.emit('bmi_update', pos=self.teleport_pos)
                self.log.info('\n')
                self.log.info('BMI Decoded Position: {}, Head-Direction: {}, Speed: {}'.format(_teleport_pos, hd, speed))
                

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
        '''
        jov emit necessary event to task by going through `task_routine` at each frame (check _jovian_process)
        One can flexibly define his/her own task_routine. 
        It provides the necessary event for the task fsm at frame rate. 
        '''
        self.cnt.add_(1)
        if self.cnt == 1:
            self.emit('start')
        # if self.cnt%2 == 0:
        self.emit('frame')
        self.check_touch_agent_to_cue()  # JUMPER, one_cue, two_cue, moving_cue etc..
        self.check_touch_cue_to_cue()    # JEDI

    def check_touch_agent_to_cue(self):
        for _cue_name in self.shared_cue_dict.keys():
            if is_close(self.current_pos, torch.tensor(self.shared_cue_dict[_cue_name]), self.touch_radius):
                self.emit('touch', args=(_cue_name, self.shared_cue_dict[_cue_name]))

    def check_touch_cue_to_cue(self):
        # here let's assume that there are only two cues to check
        _cue_name_0, _cue_name_1 = list(self.shared_cue_dict.keys())
        if is_close(torch.tensor(self.shared_cue_dict[_cue_name_0]), 
                          torch.tensor(self.shared_cue_dict[_cue_name_1]), self.touch_radius):
            self.emit('touch', args=((_cue_name_0, _cue_name_1), self.shared_cue_dict[_cue_name_0]))

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
        return self.pipe_gui_side.recv().decode("utf-8")


    def toggle_motion(self):
        cmd = "console.toggle_motion()\n"
        self.output.send(cmd.encode())


    def teleport(self, prefix, target_pos, head_direction=None, target_item=None):
        '''
           Jovian abstract (output): https://github.com/chongxi/playground/issues/6
           Core function: This is the only function that send `events` back to Jovian from interaction 
        '''
        try:
            x, y, z = target_pos # the coordination
        except:
            x, y = target_pos
            z = 0

        if head_direction is None:
            v = 0
        else:
            v = head_direction

        if prefix == 'console':  # teleport animal, target_item is None
            cmd = "{}.teleport({},{},{},{})\n".format('console', x, y, 5, v)
            self.output.send(cmd.encode())

        elif prefix == 'model':  # move cue
            with Timer('', verbose = ENABLE_PROFILER):
                z += self.shared_cue_height[target_item]
                cmd = "{}.move('{}',{},{},{})\n".format('model', target_item, x, y, z)
                self.output.send(cmd.encode())
                bottom = z - self.shared_cue_height[target_item]
                self.shared_cue_dict[target_item] = self._to_jovian_coord(np.array([x,y,bottom], dtype=np.float32))


    def reward(self, time):
        try:
            cmd = 'reward, {}'.format(time)
            self.pynq.send(cmd.encode())
        except:
            print('fail to send reward command')
