


class FPGA(object):
    """docstring for FPGA"""
    def __init__(self, arg):
        super(FPGA, self).__init__()
        self.r32 = io.open('/dev/xillybus_fet_clf_32', 'rb')
        self.r32_buf = io.BufferedReader(r32)
        # fd = os.open("./fet.bin", os.O_CREAT | os.O_WRONLY | os.O_NONBLOCK)
        # num = 0
        self._size = 7*4  # 6 samples, 4 bytes/sample

    def get_fet(self, N):
        
        