from jovian import Jovian
from task import one_cue_task, two_cue_task, one_cue_moving_task

from torch import multiprocessing
import logging
import pandas as pd
import numpy as np
from ..utils import interp_pos

_origin = np.array([-1309.17, -1258.14])
_scale  = 100.

def create_logger():
    multiprocessing.log_to_stderr()
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("process.log")
    fmt = '%(asctime)s - %(processName)s - %(levelname)s - %(funcName)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


class logger():
    def __init__(self, filename):
        text=[]
        time = []
        process = []
        level = []
        func = []
        msg = []
        SY = []
        with open(filename) as f: 
            for line in (f): 
                text.append(line)
                if line != 'SY\n':
                    asctime, processName, levelname, funcName, message = line.split(' - ')
        #             print asctime, processName, levelname, funcName, message
                    time.append(asctime.strip())
                    process.append(processName.strip())
                    level.append(levelname.strip())
                    func.append(funcName.strip())
                    msg.append(message.strip())
                if line == 'SY\n':
                    SY.append(msg[-1])
                    
        self.sync_time = int(SY[0].split(',')[0])
        self.log = pd.DataFrame(
            {'time': time,
             'process': process,
             'level': level,
             'func': func,
             'msg': msg
            })

        self.log_sessions = self.get_log_sessions()


    def get_log_sessions(self):
        log = self.log
        jov_starts = log[np.logical_and(log['process']=='MainProcess', log['msg']=='jovian_process_start')]
        jov_stops = log[np.logical_and(log['process']=='MainProcess', log['msg']=='jovian_process_stop')]
        log_sessions = [log[jov_starts.index[i]+1:jov_stops.index[i]] for i in range(jov_starts.index.shape[0])]
        return log_sessions


    def to_trajectory(self, session_id, interpolate=True, to_jovian_coord=True):
        log = self.log_sessions[session_id]
        sync_time = self.sync_time
        locs = log[log['func']=='_jovian_process']['msg'].values
        datum = np.array([[int(_) for _ in loc.replace('[','').replace(']','').split(',')] for loc in locs])
        ts = datum[:,0]
        pos = datum[:, 1:]
        if sync_time is not None:
            try:
                start_idx = np.where(ts==sync_time)[0][0]
                ts = datum[start_idx:,0]
                pos = datum[start_idx:, 1:]
                ts, pos = (ts-ts[0])/1e3, pos
            except:
                print('sync_time {} not in session {}'.format(sync_time, session_id))
                return None
        else:
            ts, pos = (ts-ts[0])/1e3, pos 

        if interpolate:
            ts, pos = interp_pos(ts, pos) 

        if to_jovian_coord is True:
            pos = pos/_scale + _origin

        return ts, pos
