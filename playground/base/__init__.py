from .jovian import Jovian
from .rotenc import Rotenc
from .task import one_cue_task, two_cue_task, one_cue_moving_task
from .fpga import Fpga
from .behaviour import interp_pos, interp_1d
from ..utils import isnotebook

from torch import multiprocessing
import logging
import re
import pandas as pd
import numpy as np
from scipy import signal
from tqdm.notebook import tqdm

_center = np.array([-1309.21, -1258.16])  # by default, will be replaced if there is maze_center in the log
_scale  = 100.  # fixed for Jovian 
float_pattern = r'([-+]?\d*\.?\d+|[-+]?\d+)' # regexp for float number

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
    def __init__(self, filename, session_id=0, sync=True):

        if isnotebook():
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        time,process,level,func,msg,SY = ([] for i in range(6))

        with open(filename) as f:
            for linenumber, line in enumerate(f):
                pass

        with open(filename) as f:
            for i in tqdm(range(linenumber)):
                line = f.readline()
                if line != 'SY\n':
                    try:
                        asctime, processName, levelname, funcName, message = line.split(' - ')
                    except:
                        pass
                    time.append(asctime.strip())
                    process.append(processName.strip())
                    level.append(levelname.strip())
                    func.append(funcName.strip())
                    msg.append(message.strip())
                if line == 'SY\n':
                    SY.append(msg[-1])

        if len(SY)==0:
            print('Critical warning: no SYNC signal found')
            sync=False

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
        self.n_sessions   = len(self.log_sessions)
        self.trial_index = None
        # print('{} sessions found'.format(self.n_sessions))

        # self.session_id  = session_id  # this will update self.df to log_sessions[session_id]


    @property
    def session_id(self):
        return self._session_id

    @session_id.setter
    def session_id(self, i):
        self._session_id = i
        self.df = self.log_sessions[self.session_id]
        print('session {} loaded into the dataframe'.format(self._session_id))


    def get_log_sessions(self):
        log = self.df
        jov_starts = log[np.logical_and(log['process']=='MainProcess', log['msg']=='jovian_process_start')]
        jov_stops = log[np.logical_and(log['process']=='MainProcess', log['msg']=='jovian_process_stop')]
        log_sessions = [log[jov_starts.index[i]+1:jov_stops.index[i]] for i in range(jov_starts.index.shape[0])]
        return log_sessions

    def to_trajectory(self, session_id, target='', interpolate=True, to_zero_center_coord=True, ball_movement=False):
        log = self.log_sessions[session_id]
        locs = log[log['func']=='_jovian_process']['msg']
        cue_flag = locs.str.contains('cue_pos')
        jov_pos = []
        cue_pos = []
        for i, loc in enumerate(locs):
            if cue_flag.iloc[i] == False:
                jov_list = loc.replace('[','').replace(']','').replace(',','').split()
                _jov_pos = [float(_) for _ in jov_list] 
                jov_pos.append(_jov_pos)
            else:
                cue1_list = loc.split(',')[1].replace('[','').replace(']','').split()
                cue2_list = loc.split(',')[2].replace('[','').replace(']','').split()
                _cue_pos = [float(_) for _ in cue1_list] + [float(_) for _ in cue2_list]
                cue_pos.append(_cue_pos)
        self.jov_pos = np.array(jov_pos)
        self.cue_pos = np.array(cue_pos)
        ts = self.jov_pos[:,0]
        pos = self.jov_pos[:, 1:3]
        z   = self.jov_pos[:, 3]
        hd = self.jov_pos[:, 4]
        cue_pos = self.cue_pos
        if ball_movement:
            ball_vel = self.jov_pos[:, 5]

        if self.sync_time is not None:
            self.start_idx = np.where(ts==self.sync_time)[0][0]
            ts  = self.jov_pos[self.start_idx:,0]
            pos = self.jov_pos[self.start_idx:, 1:]
            z   = self.jov_pos[self.start_idx:, 3]
            hd  = self.jov_pos[self.start_idx:, 4]
            if cue_pos.shape[0] > 0:
                cue_pos = self.cue_pos[self.start_idx:, :]
            if ball_movement:
                ball_vel = self.jov_pos[start_idx:, 5]
            ts, pos, cue_pos = (ts-ts[0])/1e3, pos, cue_pos
        else:
            ts, pos, cue_pos = (ts-ts[0])/1e3, pos, cue_pos 

        if interpolate:
            new_ts, pos = interp_pos(ts, pos) 
            if ball_movement:
                new_ts, ball_vel = interp_1d(ts, new_ts, ball_vel)
            ts = new_ts

        if to_zero_center_coord is True:
            pos = pos/_scale + self.maze_center  # jovian maze_center is negative
            if cue_pos.shape[0] > 0:
                cue_pos = cue_pos/_scale
                cue_pos[:, 0:2] += self.maze_center  # cue 1
                cue_pos[:, 3:5] += self.maze_center  # cue 2

        if cue_pos.shape[0] > 0:
            if ball_movement:
                return ts, pos, cue_pos, ball_vel
            else:
                return ts, pos, cue_pos
        else:
            if ball_movement:
                return ts, pos, None, ball_vel
            else:
                return ts, pos, None

    def to_pc(self, session_id=0, dt=0.3, bin_size=2.5, v_cutoff=5):
        from spiketag.analysis import place_field
        try:
            ts, pos = self.to_trajectory(session_id)
            pc = place_field(ts=ts, pos=pos, bin_size=bin_size, v_cutoff=v_cutoff, maze_range=self.maze_range)
        except:
            ts, pos, cue_pos = self.to_trajectory(session_id)
            pc = place_field(ts=ts, pos=pos, bin_size=bin_size, v_cutoff=v_cutoff, maze_range=self.maze_range)
            pc.cue_pos = cue_pos
        pc(dt)
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

    def select(self, func='', msg=''):
        df = self.df[self.df.func.str.contains(func)]
        df = df[df.msg.str.contains(msg)]
        return df

    def extractall(self, expr=r'([-+]?\d*\.?\d+|[-+]?\d+)', dtype='float', level='INFO', proc='', func='', msg=''):
        '''
        extract all matched rugular expression pattern in the msgs in defined func and msg
        Note the msg here can be a sub-string (but need to be continuous) to be pattern complete
        '''
        df = self.df[self.df.process.str.contains(proc)]
        df = df[df.level.str.contains(level)]
        df = df[df.func.str.contains(func)]
        df = df[df.msg.str.contains(msg)]
        return df.msg.str.extractall(expr).astype(dtype).unstack().to_numpy()
        
    def get_trial_index(self, start_with='BMI control enabled', end_with='BMI control disabled'):
        '''
        check https://github.com/chongxi/playground/issues/24

        get trial index from the process.log file
        each trial begins with 'BMI control enabled' in its msg field
        each trial end with 'BMI control disabled' in its msg field
        Note:
            both msg are sent by the same process (jovian) 
            each msg via different function defined in task (enabled by `reset`, disabled by `goal_cue_touched`)
        '''
        bmi_enable = self.select(func='', msg=start_with)
        bmi_disable = self.select(func='', msg=end_with)
        df = pd.concat([bmi_enable, bmi_disable]).sort_index()
        if len(df) % 2 == 1:
            self.trial_index = df.index.to_numpy()[:-1].reshape(-1, 2)
            return self.trial_index
        else:
            self.trial_index = df.index.to_numpy().reshape(-1, 2)
            return self.trial_index

    @property
    def trial_df_orig(self):
        if self.trial_index is None:
            index = self.get_trial_index()
        else:
            index = self.trial_index
        trial_df = []
        for i in range(len(index)):
            _df = self.df.loc[index[i,0]:index[i,1]]
            trial_df.append(_df)
        return trial_df

    def get_epoch_non_bmi(self, i, trial_index=None):
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
        epoch_df = epoch_df[~epoch_df.msg.str.contains('cue_pos')]

        # 2. get bmi_pos and goal_pos
        # bmi_pos = bmi_df.loc[:,['x','y']].to_numpy()
        cue_pos = np.array([float(_) for _ in re.findall("\d+\.", epoch_df.iloc[-2].msg)])[:2]
        goal_pos = cue_pos/_scale + self.maze_center

        # 3. get the jovian time and jovian ball vellocity for this epoch
        jov_df = epoch_df[epoch_df['func']=='_jovian_process'].msg.str.extractall(float_pattern).astype('float').unstack()
        jov_df.columns = ['time', 'x','y','z','v','ball_v']
        jov_time = np.array(jov_df.loc[jov_df.index[0]:jov_df.index[-1]].time.to_numpy())/1e3
        jov_time -= jov_time[0]
        epoch_time = jov_time[-1]
        # ball_vel = jov_df.ball_v.mean()

        return epoch_time, goal_pos, jov_df

        
    def get_epoch_bmi(self, i, bypass_bmi_outlier=True, trial_index=None):
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


    def _repr_html_(self):
        return self.df._repr_html_()
