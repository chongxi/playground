from .jovian import Jovian
from .rotenc import Rotenc
from .task import one_cue_task, two_cue_task, one_cue_moving_task
from .fpga import Fpga
from .behaviour import interp_pos, interp_1d

from torch import multiprocessing
import logging
import re
import pandas as pd
import numpy as np
from scipy import signal

_center = np.array([-1309.21, -1258.16])  # by default, will be replaced if there is maze_center in the log
_scale  = 100.  # fixed for Jovian 
float_pattern = r'([-+]?\d*\.\d+|\d+)' # regexp for float number

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
    def __init__(self, filename, sync=True):
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
                    try:
                        asctime, processName, levelname, funcName, message = line.split(' - ')
                    except:
                        print('{} cannot be parsed'.format(line))
        #             print asctime, processName, levelname, funcName, message
                    time.append(asctime.strip())
                    process.append(processName.strip())
                    level.append(levelname.strip())
                    func.append(funcName.strip())
                    msg.append(message.strip())
                if line == 'SY\n':
                    SY.append(msg[-1])

        if sync:
            self.sync_time = int(SY[0].split(',')[0])
        else:
            self.sync_time = None

        self.df = pd.DataFrame(
            {'time': time,
             'process': process,
             'level': level,
             'func': func,
             'msg': msg
            })

        self.log_sessions = self.get_log_sessions()

    def get_log_sessions(self):
        log = self.df
        jov_starts = log[np.logical_and(log['process']=='MainProcess', log['msg']=='jovian_process_start')]
        jov_stops = log[np.logical_and(log['process']=='MainProcess', log['msg']=='jovian_process_stop')]
        log_sessions = [log[jov_starts.index[i]+1:jov_stops.index[i]] for i in range(jov_starts.index.shape[0])]
        return log_sessions

    def to_trajectory(self, session_id, target='', interpolate=True, to_zero_center_coord=True, ball_movement=False):
        log = self.log_sessions[session_id]
        locs = log[log['func']=='_jovian_process']['msg'].values
        # datum = np.array([[int(_) for _ in loc.replace('[','').replace(']','').split(',')] for loc in locs])
        datum = []
        for i, loc in enumerate(locs):
            #dd = loc.replace('[','').replace(']','').split(',')
            dd = loc.replace('[','').replace(']','').replace(',','').split()
            if len(dd[1:])<3:
                print(loc)
            else:
                datum.append([float(_) for _ in dd])
        self.datum = np.array(datum)
        ts = self.datum[:,0]
        pos = self.datum[:, 1:3]
        z   = self.datum[:, 3]
        hd = self.datum[:, 4]
        if ball_movement:
            ball_vel = self.datum[:, 5]

        if self.sync_time is not None:
            start_idx = np.where(ts==self.sync_time)[0][0]
            ts  = self.datum[start_idx:,0]
            pos = self.datum[start_idx:, 1:]
            z   = self.datum[start_idx:, 3]
            hd  = self.datum[start_idx:, 4]
            if ball_movement:
                ball_vel = self.datum[start_idx:, 5]
            ts, pos = (ts-ts[0])/1e3, pos
        else:
            ts, pos = (ts-ts[0])/1e3, pos 

        if interpolate:
            new_ts, pos = interp_pos(ts, pos) 
            if ball_movement:
                new_ts, ball_vel = interp_1d(ts, new_ts, ball_vel)
            ts = new_ts

        if to_zero_center_coord is True:
            pos = pos/_scale + self.maze_center  # jovian maze_center is negative

        if ball_movement:
            return ts, pos, ball_vel
        else:
            return ts, pos


    def to_pc(self, session_id=0):
        from spiketag.analysis import place_field
        ts, pos = log.to_trajectory(session_id)
        pc = place_field(ts=ts, pos=pos, maze_range=self.maze_range)
        return pc


    @property
    def maze_center(self):
        try:
            sub_df = self.df[self.df['func']=='load_maze']
            _maze_center = sub_df[sub_df.msg.str.contains('maze_center')].msg.str.extractall(r'([-+]?\d*\.\d+|\d+)').astype('float').unstack().iloc[0].to_numpy()
            return _maze_center
        except:
            return _center
            print('check whether maze_center is in the log')

    @property
    def maze_range(self):
        try:
            sub_df = self.df[self.df.func=='load_maze']
            s = sub_df[sub_df.msg.str.contains('maze_border')].iloc[0].msg.split('[')[-1].replace(']','').split(' ')
            maze_range = [float(_) for _ in s if _ is not '']
            maze_range = np.array(maze_range)
            return maze_range.reshape(-1,2).T
        except:
            print('check whether maze_border is in the log')

    def get_trial_index(self, start_with='parachute finished', end_with='touch'):
        '''
        check https://github.com/chongxi/playground/issues/24

        with trial index, try:
        i = trial_to_check
        log.df.iloc[trial_index[i,0]-1:trial_index[i,1]]
        '''
        trial_start = self.df[(self.df['func']=='on_animation_finish') & (self.df['msg'].str.contains(start_with))].index[1:] + 1
        trial_end   = self.df[(self.df['func']=='on_event') & (self.df['msg'].str.contains(end_with))].index[1:] + 1
        n_trials    = min(trial_start.to_numpy().shape[0], trial_end.to_numpy().shape[0])
        if start_with == 'parachute finished':
            trial_index = np.zeros((n_trials, 2), dtype='int')
            trial_index[:,0] = trial_start[:n_trials].to_numpy()
            trial_index[:,1] = trial_end[:n_trials].to_numpy()
        elif start_with == 'bury finished':
            trial_index = np.zeros((n_trials-1, 2), dtype='int')
            trial_index[:,0] = trial_start[:n_trials-1].to_numpy()
            trial_index[:,1] = trial_end[1:n_trials].to_numpy()
        return trial_index
        
        
    def get_epoch(self, i, bypass_bmi_outlier=True, trial_index=None):
        '''
        get varialbes of `i`th trial
        
        To further use the bmi_df and jov_df:
        bmi_pos = bmi_df.loc[:,['x','y']].to_numpy()
        jov_pos = jov_df.loc[:,['x','y']].to_numpy()/100 + log.maze_center
        ball_vel = jov_df.ball_v.mean()
        '''
        if trial_index is None:
            trial_index = self.get_trial_index(start_with='parachute finished', end_with='touch')

        # 1. get epoch dataframe
        epoch_df = self.df.iloc[trial_index[i,0]-1:trial_index[i,1]+1]
        bmi_df = epoch_df[epoch_df['func']=='on_decode'].msg.str.extractall(float_pattern).astype('float').unstack()
        bmi_df.columns = ['x','y','ball_thres']

        if bmi_df.shape[0]<=2:
            print('not enough bmi teleportation points')
            return [None]*5

        # 2. get bmi_pos and goal_pos
        # bmi_pos = bmi_df.loc[:,['x','y']].to_numpy()
        cue_pos = np.array([float(_) for _ in re.findall("\d+\.", epoch_df.iloc[-2].msg)])[:2]
        goal_pos = cue_pos/100 + self.maze_center

        # 3. get the jovian time and jovian ball vellocity for this epoch
        jov_df = epoch_df[epoch_df['func']=='_jovian_process'].msg.str.extractall(float_pattern).astype('float').unstack()
        jov_df.columns = ['time', 'x','y','z','v','ball_v']
        jov_time = np.array(jov_df.loc[bmi_df.index[0]:bmi_df.index[-1]].time.to_numpy())/1e3
        jov_time -= jov_time[0]
        epoch_time = jov_time[-1]
        # ball_vel = jov_df.ball_v.mean()

        # 4. get landing position
        start_pos = bmi_df[bmi_df.index > epoch_df[epoch_df['msg']=='parachute finished'].index.to_numpy()[0]+1].iloc[0][['x','y']].to_numpy()

        return epoch_time, start_pos, goal_pos, bmi_df, jov_df
