import serial
import numpy as np
import torch as torch
from torch.multiprocessing import Process

class Rotenc(object):
    """
    Rotation Encoder: from 0.0-360.0
    read the continous stream like
    ...
    b'328.37\r\n'
    b'327.87\r\n'
    ...
    from serial port serial.Serial('/dev/ttyACM0', 192000)
    """

    def __init__(self, dev='/dev/ttyACM0', baudrate=192000):
        self.dev = dev
        try:
            self.ser = serial.Serial(dev, baudrate)
            self.is_connected = True
        except:
            print('cannot find serial port at {} to read rotation degree'.format(self.dev))
            self.is_connected = False
        self.direction = torch.empty(1,)
        self.direction.share_memory_()
        self.direction.fill_(0.0)

    def _rotenc_process(self):
        while self.is_connected:
            self.direction.fill_(float(self.ser.readline().decode("utf-8")))
            # print(self.direction)

    def start(self):
        self.rotenc_process = Process(target=self._rotenc_process)
        self.rotenc_process.daemon = True
        self.rotenc_process.start()  

    def stop(self):
        self.rotenc_process.terminate()
        self.rotenc_process.join()

