import numpy as np
from scipy.interpolate import interp1d

def interp_pos(t, pos, N=1):
    '''
    convert irregularly sampled pos into regularly sampled pos
    N is the dilution sampling factor. N=2 means half of the resampled pos
    '''
    dt = np.mean(np.diff(t))
    x, y = interp1d(t, pos[:,0], fill_value="extrapolate"), interp1d(t, pos[:,1], fill_value="extrapolate")
    new_t = np.arange(0.0, dt*len(t), dt*N)
    new_pos = np.hstack((x(new_t).reshape(-1,1), y(new_t).reshape(-1,1)))
    return new_t, new_pos    

def interp_1d(t_old, t_new, var):
    '''
    interp_1d variable
    '''
    x = interp1d(t_old, var, fill_value="extrapolate")
    new_var = x(t_new)
    return t_new, new_var    


