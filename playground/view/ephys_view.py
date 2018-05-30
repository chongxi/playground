import numpy as np
from spiketag.base import probe
from spiketag.view import probe_view



if __name__ == '__main__':

    prb = probe(shank_no=3)

    prb.shanks[0].l = [59,60,10,58,12,11,57,56]
    prb.shanks[0].r = [5,52,3,54,53,4,13,2,55]
    prb.shanks[0].xl = -100.
    prb.shanks[0].yl = 20
    prb.shanks[0].xr = -80.
    prb.shanks[0].yr = 5

    prb.shanks[1].l = [15,63,48,47,0,61,9,14,62,6]
    prb.shanks[1].r = [8, 1,51,50,18,34,31,25,33,17,22,49]
    prb.shanks[1].xl = -10.
    prb.shanks[1].yl = 15
    prb.shanks[1].xr = 10.
    prb.shanks[1].yr = 0 

    prb.shanks[2].l = [39,38,20,45,44,24,7,32,16,23,46,30]
    prb.shanks[2].r = [19,37,21,35,36,26,29,40,27,42,41,28,43]
    prb.shanks[2].xl = 80.
    prb.shanks[2].yl = 10 
    prb.shanks[2].xr = 100.
    prb.shanks[2].yr = -5
    prb.auto_pos()
    prb.mapping[5]  += np.array([-10,2])
    prb.mapping[52] += np.array([-2, 0])
    prb.mapping[8]  += np.array([-10,2])
    prb.mapping[1]  += np.array([-2, 0])
    prb.mapping[19] += np.array([-10,2])
    prb.mapping[37] += np.array([-2, 0])
    # print prb.mapping
    prb.grp_dict = {0: np.array([60,59,52,5]), 1:np.array([39,19,37,21])}
    prb[2] = np.array([62,17,22,6])
    print prb

    prb_view = probe_view()
    prb_view.set_data(prb)
    prb_view.run()
