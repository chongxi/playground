import os 
import io
import time
import numpy as np
# from spiketag.view.grid_scatter3d import grid_scatter3d

'''
through `/dev/xillybus_fet_clf_32` each spike will generate 7 datum:
{0:time, 1:grpNo, 2:fet0, 3:fet1, 4:fet2, 5:fet3,  6:1nn_id}
'''

def get_fet(info=False, vis=False):
    '''
    A daemon process dedicated on reading data from PCIE and update
    the shared memory with other processors: shared_arr 
    '''
    # r32 = pcie_recv_open()
    r32 = io.open('/dev/xillybus_fet_clf_32', 'rb')
    r32_buf = io.BufferedReader(r32)
    fd = os.open("./fet.bin", os.O_CREAT | os.O_WRONLY | os.O_NONBLOCK)
    num = 0
    _size = 7*4  # 6 samples, 4 bytes/sample


#     if vis == True:
        # rows, cols = 6, 8
        # win = grid_scatter3d(rows, cols)
        # win.show()

    
    while True:
        # with shared_arr.get_lock():
        tic = time.time() * 1000
        # buf = r32_buf.read(_size)
        buf = r32.read(_size)
        # f.write(buf)
        os.write(fd, buf)
        toc = time.time() * 1000
        print '{0} ms'.format(toc-tic)
        
        if info == True:
            fet = np.frombuffer(buf,dtype=np.int32).reshape(-1,7)
            fet_info = fet[:,:2]
            print fet_info

if __name__ == "__main__":
    get_fet(info=True, vis=True)
