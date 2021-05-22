import sys
import socket
import numpy as np
import torch as torch
from torch.multiprocessing import Process, Pipe
from spiketag.utils import Timer
from spiketag.utils import EventEmitter
from spiketag.analysis.core import get_hd
from spiketag.utils import FIFO
from ..rotenc import Rotenc


ENABLE_PROFILER = False

# Lab
host_ip = '10.102.20.30'
pynq_ip = '10.102.20.75'


# Test
# host_ip = '10.102.20.42'
# pynq_ip = '127.0.0.1'
# verbose = True

is_close = lambda pos, cue_pos, radius: (pos-cue_pos).norm()/100 < radius


class Jovian_Stream(str):
    def parse(self):
        line = self.__str__()
        _line = line.split(',')
        try:
            _t,_x,_y,_ball_vel = int(_line[0]), int(_line[1]), int(_line[2]), int(_line[7])
            _coord = [_x, _y, 0]
            return _t, _coord, _ball_vel
        except:
            _t,_info = int(_line[0]), _line[1]
            return _t, _info

        
def rotate(pos, theta=0):
    x = theta/360*2*np.pi
    R = np.array([[cos(x), -sin(x)], [sin(x), cos(x)]])
    return R.dot(pos)


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
        self.rotenc_init()


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
        try:
            self.pynq = socket.create_connection((pynq_ip, '2222'), timeout=1)
            self.pynq.setblocking(1)
            self.pynq.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.pynq_connected = True
            self.socks = [self.input,  self.output, self.output_control, self.pynq]
        except:
            self.pynq_connected = False
            self.socks = [self.input,  self.output, self.output_control]


    def rotenc_init(self):
        '''
        init the rotenc
        '''
        self.rot = Rotenc()


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
        self.hd_window.fill_(0)
        self.ball_vel = torch.empty(1,)
        self.ball_vel.share_memory_()
        self.ball_vel.fill_(0)
        self.bmi_hd = torch.empty(1,)       # calculated hd sent to Jovian for VR rendering
        self.bmi_hd.share_memory_()
        self.bmi_hd.fill_(0)         
        self.current_hd = torch.empty(1,)   # calculated hd (same as bmi_hd) sent to Mazeview for local playground rendering
        self.current_hd.share_memory_()
        self.current_hd.fill_(0)

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
                    self._t, self._coord, self._ball_vel = self.readline().parse()
                    _cue_name_0, _cue_name_1 = list(self.shared_cue_dict.keys())
                    if type(self._coord) is list:
                        self.current_pos[:]  = torch.tensor(self._coord)
                        self.current_hd[:]   = self.rot.direction
                        self.ball_vel[:]     = self._ball_vel
                        self.log.info('{}, {}, {}, {}'.format(self._t, self.current_pos.numpy(), 
                                                              self.current_hd.numpy(), 
                                                              self._ball_vel))
                        self.log.info('cue_pos:, {},{}'.format(self.shared_cue_dict[_cue_name_0],
                                                               self.shared_cue_dict[_cue_name_1]))
                        self.task_routine()
                    else:
                        self.log.warn('{}, {}'.format(self._t, self._coord))

                except:
                    self.log.info('socket time out')


    def set_bmi(self, bmi, pos_buffer_len=10):
        '''
        This set BMI, Its binner and decoder event for JOV to act on. The event flow:
        bmi.binner.emit('decode', X) ==> jov
        customize the post decoding calculation inside the function
        `on_decode(X)` where the X is sent from the bmi.binner, but the `self` here is the jov

        set_bmi connect the event flow from
                 decode(X)                shared variable
             y=dec.predict_rt(X)         (bmi_pos, bmi_hd)
        bmi =====================> jov ====================> task
        '''
        ## Set the BMI buffer for smoothing both pos and hd
        self.bmi = bmi
        self.bmi_pos_buf = np.zeros((pos_buffer_len, 2))
        hd_buffer_len = int(self.hd_window.item()/self.bmi.binner.bin_size)
        self.bmi_hd_buf  = np.zeros((hd_buffer_len, 2))
        self.bmi_hd_buf_ring = np.zeros((hd_buffer_len, ))
        self.log.info('Initiate the BMI decoder and playground jov connection')
        self.log.info('position buffer length:{}'.format(pos_buffer_len))

        ## Set the real-time posterior placehodler
        dumb_X = np.zeros((self.bmi.binner.B, self.bmi.binner.N-1))
        self.perm_idx = np.random.permutation(dumb_X.shape[1])
        _, post_2d = self.bmi.dec.predict_rt(dumb_X)
        self.current_post_2d = torch.empty(post_2d.shape)
        self.current_post_2d.share_memory_()
        self.log.info('The decoder binsize:{}, the B_bins:{}'.format(self.bmi.binner.bin_size, self.bmi.binner.B))
        self.log.info('The decoder input (spike count bin) shape:{}'.format(dumb_X.shape))
        self.log.info('The decoder output (posterior) shape: {}'.format(self.current_post_2d.shape))
        self.log.info('The bmi position update rule: {}'.format(self.bmi.bmi_update_rule))

        self.speed_fifo = FIFO(depth=39)
        # self.bmi.dec.drop_neuron(np.array([7,9]))

        @self.bmi.binner.connect
        def on_decode(X):
            '''
            This event is triggered every time a new bin is filled (based on BMI output timestamp)
            '''
            # print(self.binner.nbins, self.binner.count_vec.shape, X.shape, np.sum(X))
            with Timer('decoding', verbose=False):
                # ----------------------------------
                # 1. Ring decoder for the head direction
                # ----------------------------------
                # hd = self.bmi.dec.predict_rt(X) # hd should be a angle from [0, 360]
                # self.bmi_hd_buf_ring = np.hstack((self.bmi_hd_buf_ring[1:], hd))
                # # print(self.bmi_hd_buf_ring)
                # self.bmi_hd[:] = torch.tensor(self.bmi_hd_buf_ring.mean()) 

                # ----------------------------------
                # 2. Bayesian decoder for the position
                # ----------------------------------
                # if X.sum(axis=0)>2:
                # _X = X[:, self.perm_idx]
                y, post_2d = self.bmi.dec.predict_rt(X)
                post_2d /= post_2d.sum()
                max_posterior = post_2d.max()
                
                ### save scv and posterior to file ###
                f_scv = open('./scv.bin', 'ab+')
                f_scv.write(X.tobytes())
                f_scv.close()
                
                f_post = open('./post_2d.bin', 'ab+')
                f_post.write(post_2d.tobytes())
                f_post.close()
                
                ### Key: filter out criterion ###
                if X.sum()>2:
                    self.current_post_2d[:] = torch.tensor(post_2d) * 1.0
                    
                # #################### just for dusty test #########################
                # y += np.array([263.755, 263.755])
                # y -= np.array([253.755, 253.755])
                # y -= np.array([318.529, 195.760])
                # y /= 4.5
                # ##################################################################
                ball_vel_thres = self.bmi_teleport_radius.item()
                self.speed_fifo.input(self.ball_vel.numpy()[0])
                # self.log.info('FIFO:{}'.format(self.speed_fifo.numpy()))
                speed = self.speed_fifo.mean()/14e-3/100
                self.log.info('speed:{}, threshold:{}'.format(speed, ball_vel_thres))
                self.log.info('max_post:{}, post_thres:{}'.format(max_posterior, self.bmi.posterior_threshold))
                # current_speed = self.speed_fifo.mean()
                try:
                    if self.bmi.bmi_update_rule == 'moving_average':
                        # # rule1: decide the VR output by FIFO smoothing
                        if speed < ball_vel_thres and X.sum()>2 and max_posterior>self.bmi.posterior_threshold:
                            self.bmi_pos_buf = np.vstack((self.bmi_pos_buf[1:, :], y))
                            _teleport_pos = np.mean(self.bmi_pos_buf, axis=0)
                            self.log.info('_teleport_pos:{}'.format(_teleport_pos))
                        else:
                            _teleport_pos = self.bmi_pos.numpy()
                    elif self.bmi.bmi_update_rule == 'fixed_length':
                        # # rule2: decide the VR output by fixed length update
                        u = (y-self.bmi_pos.numpy())/np.linalg.norm(y-self.bmi_pos.numpy())
                        tao = 5
                        if speed < ball_vel_thres and X.sum()>2 and max_posterior>self.bmi.posterior_threshold:
                            tao = 5 # cm
                            _teleport_pos = self.bmi_pos.numpy() + tao*u 
                        else:
                            _teleport_pos = self.bmi_pos.numpy()
                    # # set shared variable
                    _teleport_pos = rotate(_teleport_pos, theta=0)
                    self.bmi_pos[:] = torch.tensor(_teleport_pos)

                    # self.bmi_hd_buf = np.vstack((self.bmi_hd_buf[1:, :], _teleport_pos))
                    # window_size = int(self.hd_window[0]/self.bmi.binner.bin_size)
                    # hd, speed = get_hd(trajectory=self.bmi_hd_buf[-window_size:], speed_threshold=0.6, offset_hd=0)
                        # hd = 90
                        # if speed > .6:
                            # self.bmi_hd[:] = torch.tensor(hd)      # sent to Jovian
                            # self.current_hd[:] = torch.tensor(hd)  # sent to Mazeview
                        # self.emit('bmi_update', pos=self.teleport_pos)
                        # self.log.info('\n')

                    self.log.info('BMI output(x,y,speed,ball_thres): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}'.format(_teleport_pos[0],
                                                                                                                _teleport_pos[1], 
                                                                                                                speed, 
                                                                                                                ball_vel_thres))
                except Exception as e:
                    self.log.warn('BMI error: {}'.format(e))
                    pass


                    

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
                self.emit('touch', args=( _cue_name, self.shared_cue_dict[_cue_name] ))

    def check_touch_cue_to_cue(self):
        # here let's assume that there are only two cues to check
        _cue_name_0, _cue_name_1 = list(self.shared_cue_dict.keys())
        if is_close(torch.tensor(self.shared_cue_dict[_cue_name_0]), 
                          torch.tensor(self.shared_cue_dict[_cue_name_1]), self.touch_radius):
            self.emit('touch', args=( _cue_name_0 + '->' + _cue_name_1, self.shared_cue_dict[_cue_name_0] ))

    def start(self):
        self.rot.start()
        self.pipe_jovian_side, self.pipe_gui_side = Pipe()
        self.jovian_process = Process(target=self._jovian_process, name='jovian') #, args=(self.pipe_jovian_side,)
        self.jovian_process.daemon = True
        self.reset() # !!! reset immediately before start solve the first time jam issue
        self.jovian_process.start()  

    def stop(self):
        self.jovian_process.terminate()
        self.jovian_process.join()
        self.cnt.fill_(0)
        self.rot.stop()

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


    def move_to(self, x, y, z=5, hd=0, hd_offset=0): 
        '''
        x,y = 0,0 # goes to the center (Jovian protocol)
        hd_offset = jov.rot.direction # the body direction
        '''
        cmd="{}.teleport({},{},{},{})\n".format('console', x, y, z, hd+hd_offset) 
        self.output.send(cmd.encode()) 


    def reward(self, time):
        try:
            cmd = 'reward, {}'.format(time)
            self.pynq.send(cmd.encode())
        except:
            print('fail to send reward command, pynq connected: {}'.format(self.pynq_connected))
