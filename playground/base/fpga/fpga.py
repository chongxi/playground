import io
import os
import time

import sys
import socket
import numpy as np
import torch as torch
from torch.multiprocessing import Process, Pipe
from ...utils import Timer
from ...utils import EventEmitter



class Fpga(object):
    """docstring for FPGA"""
    def __init__(self, prb):
        self.prb = prb
        self.group_idx = np.array(self.prb.grp_dict.keys())
        self.reset()


    def close(self):
        self.r32.close()


    def reset(self):
        self.r32 = io.open('/dev/xillybus_fet_clf_32', 'rb')
        # self.r32_buf = io.BufferedReader(r32)
        self.fd = os.open("./fet.bin", os.O_CREAT | os.O_WRONLY | os.O_NONBLOCK)
        self._size = 7*4  # 6 samples, 4 bytes/sample
        self.shared_mem_init()


    def shared_mem_init(self):
        n_spike_count_vector = len(self.prb.grp_dict.keys())
        # trigger task using frame counter
        self.spike_count_vector = torch.zeros(n_spike_count_vector,)
        self.spike_count_vector.share_memory_()


    def _fpga_process(self):
        '''
        A daemon process dedicated on reading data from PCIE and update
        the shared memory with other processors: shared_arr 
        '''
        
        tic = time.time() * 1000
        while True:
            # with shared_arr.get_lock():
            # tic = time.time() * 1000
            # buf = r32_buf.read(_size)
            buf = self.r32.read(self._size)
            # f.write(buf)
            os.write(self.fd, buf)
            toc = time.time() * 1000
            # self.log.info('{} elapsed'.format(toc-tic))
            # print '{0} ms'.format(toc-tic)
            # if info == True:
            fet = np.frombuffer(buf,dtype=np.int32).reshape(-1,7)
            fet_info = fet[:,:2]
            for group_id in fet_info[:,1]:
                if group_id in self.group_idx:
                    self.spike_count_vector[group_id] += 1
                    self.log.info('{}'.format(self.spike_count_vector.numpy()))
            # _unique, _counts =  np.unique(fet_info[:,1], return_counts=True) 
            # for i in _unique:
            #     if i in self.group_idx:
            #         self.log.info('{}: {}'.format(i, _counts[i]))
            #         self.spike_count_vector[self.prb[i]] += torch.tensor(_counts[i])


            # self.log.info('{}'.format(fet_info[0]))


    def start(self):
        self.fpga_process = Process(target=self._fpga_process, name='fpga') #, args=(self.pipe_jovian_side,)
        self.fpga_process.daemon = True
        self.fpga_process.start()  


    def stop(self):
        self.fpga_process.terminate()
        self.fpga_process.join()

