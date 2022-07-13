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
    def __init__(self, filename):
        """Load playground log file and extract BMI/Jovian/Sync data.

        Args:
            filename (string): path to the log file ('./process.log')
        
        Important Variables:
            logger.df (pandas.DataFrame): log dataframe
            logger.sync_df (pandas.DataFrame): the jovian output that exactly labelled by the sync time
            logger.jov_df (pandas.DataFrame): the animal jovian data content (time, [pos_x, pos_y], [head_direction], ball_vel)
            logger.cue_df (pandas.DataFrame): the cue jovian data content

        Other variables:
            logger.n_sessions (int): number of sessions (usually just 1, there can be bugs with multiple sessions)
            logger.log_sessions (list): list of log sessions, each is a dataframe
            logger.sync_time (int): sync time calculated by a microcontroller that synced with jovian
        """
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
                    sync_time_index = i
                    time.append(time[-1])
                    process.append('SYNC')
                    level.append('INFO')
                    func.append('sync')
                    msg.append('last message is synced')

        print(f'Creating major data frame log.df and severl sub-dataframes', end='...')
        self.df = pd.DataFrame(
            {'time': time,
             'process': process,
             'level': level,
             'func': func,
             'msg': msg
            })

        self.cue_df = self.select(func='_jovian', msg='cue_pos')
        self.cue_idx = self.cue_df.index  # index of cue position data in the log dataframe (report by _jovian process in playground)
        self.jov_df = self.select(func='_jovian').drop(self.cue_df.index)
        self.jov_idx = self.jov_df.index  # index of jovian animal position data in the log dataframe (report by _jovian process in playground)
        self.reward_df = self.select(func='touched', msg='reward')
        self.touch_df = self.select(func='', msg='touch:')
        print('Done')

        if len(SY) == 0:
            print('Critical warning: no SYNC signal found')
            self.sync_time = None
        else:
            print('Find SYNC, syncing the data', end='...')
            self.sync_time = int(SY[0].split(',')[0])
            self.sync_idx = self.select(func='sync').index - 1 # index of jov timestamps that is exactly same as SY
            self.sync_df = self.df.loc[self.sync_idx]
            self.jov_idx = self.jov_idx[self.jov_idx >= self.sync_idx[0]]  # only use jovian data after the sync time
            self.jov_df = self.jov_df.loc[self.jov_idx]
            self.cue_idx = self.cue_idx[self.cue_idx > self.jov_idx[0]]    # only use cue data after the first jovian data
            self.cue_df = self.cue_df.loc[self.cue_idx]
            self.reward_df = self.reward_df[self.reward_df.index > self.jov_idx[0]]
            self.touch_df = self.touch_df[self.touch_df.index > self.jov_idx[0]]
            print('Done')

        print(f'Finalizing all sub-dataframes', end='...')
        self.jov_pos_df = self.jov_df.msg.str.extractall(float_pattern).unstack().astype('float')
        self.jov_pos_df.columns = ['jov_time', 'jov_x', 'jov_y', 'jov_z', 'jov_hd', 'jov_ball_vel']
        self.cue_pos_df = self.cue_df.msg.str.extractall(float_pattern).unstack().astype('float')
        self.cue_pos_df.columns = ['cue1_x', 'cue1_y', 'cue1_z', 'cue2_x', 'cue2_y', 'cue2_z']

        self.dfs = {'jov_df': self.jov_df,
                    'jov_pos_df': self.jov_pos_df,
                    'cue_df': self.cue_df,
                    'reward_df': self.reward_df,
                    'touch_df': self.touch_df}
        print('Done')
        print('Please check log.df, log.jov_pos_df, log.cue_df, log.reward_df, log.touch_df')

        self.log_sessions = self.get_log_sessions()
        self.n_sessions   = len(self.log_sessions)
        self.trial_index = None
        # print('{} sessions found'.format(self.n_sessions))


    @property
    def session_id(self):
        return self._session_id

    @session_id.setter
    def session_id(self, i):
        self._session_id = i
        self.df = self.log_sessions[self.session_id]
        print('session {} loaded into the dataframe'.format(self._session_id))

    def total_sedonds(self, df):
        return (pd.to_datetime(df.time.iloc[-1]) - pd.to_datetime(df.time.iloc[0])).total_seconds()

    def get_log_sessions(self):
        log = self.df
        jov_starts = log[np.logical_and(log['process']=='MainProcess', log['msg']=='jovian_process_start')]
        jov_stops = log[np.logical_and(log['process']=='MainProcess', log['msg']=='jovian_process_stop')]
        log_sessions = [log[jov_starts.index[i]+1:jov_stops.index[i]] for i in range(jov_starts.index.shape[0])]
        return log_sessions

    def get_jov(self):
        """
        get jovian datastream from the log sub-dataframes:
                jov_pos_df: jovian main data stream
                cue_pos_df: cue position data stream
                reward_df: reward data stream
                touch_df: touch data stream (currently not used here)
        Returns:
            (ts, pos, hd, ball_vel, cue_pos, reward_time): tuple of numpy arrays
        """
        jov = self.jov_pos_df.to_numpy()
        t = jov[:, 0]
        self._jov_ts = (t - t[0])/1e3
        self._jov_pos = self.convert_jov_pos(jov[:, 1:3])
        self._jov_hd = jov[:, -2]
        self._jov_ball_vel = jov[:, -1]
        self._jov_reward_time = self.jov_pos_df.iloc[self.jov_pos_df.index.searchsorted(self.reward_df.index)-1].jov_time.to_numpy()
        self._jov_reward_time = (self._jov_reward_time - t[0])/1e3
        cue_pos = self.cue_pos_df.to_numpy()
        cue_pos[:, 0:2] = self.convert_jov_pos(cue_pos[:, :2])
        cue_pos[:, 3:5] = self.convert_jov_pos(cue_pos[:, 3:5])
        self._jov_cue_pos = cue_pos
        jov_dict =  {'jov_ts': self._jov_ts, 
                     'jov_pos': self._jov_pos, 
                     'jov_hd': self._jov_hd,
                     'jov_ball_vell': self._jov_ball_vel, 
                     'jov_cue_pos': self._jov_cue_pos, 
                     'jov_reward_time': self._jov_reward_time}
        return jov_dict

    def to_trajectory(self, session_id=0, target='', interpolate=True, to_zero_center_coord=True, ball_movement=False):
        """
        deprecated soon, use `get_jov` for more complete read out with much faster speed
        """
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
                ball_vel = self.jov_pos[self.start_idx:, 5]
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

    def to_pc(self, session_id=0, dt=0.1, bin_size=2.5, v_cutoff=5):
        from spiketag.analysis import place_field
        jov_dict = self.get_jov()
        ts, pos, cue_pos = jov_dict['jov_ts'], jov_dict['jov_pos'], jov_dict['jov_cue_pos']
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
        
    @property
    def bmi_params(self):
        '''
        The number of cell count, bin window length and decoding window length
        '''
        cell_count, bin_len, dec_len = self.select(func='build_decoder', msg='params').msg.str.extractall(
                                                      float_pattern).astype('float').unstack().to_numpy().ravel()
        return int(cell_count), bin_len, dec_len

    @property
    def task_params(self):
        '''
        The task parameters
        '''
        reward_time, reward_radius = self.select(func='Task', msg='radius').msg.str.extractall(
                                                    float_pattern).unstack().to_numpy().astype('float').ravel()
        return reward_time, reward_radius

    @property
    def bmi_pos_vel(self):
        '''
        The full BMI decoded matrix: 
        Four columns: bmi_x, bmi_y, ball_vel, ball_vel_threshold
        '''
        bmi_pos = self.extractall(func='on_decode', msg='output')
        return bmi_pos
    
    def convert_jov_pos(self, pos):
        jov_pos = pos/_scale + self.maze_center
        return np.round(jov_pos, 2) + 0.01

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
    
    def get_jov_after_bmi(self, trial_no=None):
        trial_df = self.trial_df_orig[trial_no]
        trial_bmi_idx = trial_df[trial_df.func.str.contains('decode') & trial_df.msg.str.contains('BMI')].index
        _jov_df = self.df.loc[self.jov_idx[np.searchsorted(self.jov_idx, trial_bmi_idx.to_numpy())]]
        
        jov_pos_df = _jov_df.reset_index().msg.str.extractall(float_pattern).astype('float').unstack() 
        jov_pos = jov_pos_df.to_numpy()[:, 1:3]
        jov_pos = self.convert_jov_pos(jov_pos) # convert to the maze coordinate
        jov_hd = jov_pos_df.to_numpy()[:, 4]
        jov_ball_vel = jov_pos_df.to_numpy()[:, -1]
        return jov_pos, jov_hd, jov_ball_vel

    def get_cue_pos(self, trial_no):
        trial_df = self.trial_df_orig[trial_no]
        cue_pos = trial_df[trial_df.func.str.contains('on_event') & trial_df.msg.str.contains('touch')].msg.str.extractall(
                            float_pattern).unstack().to_numpy().astype('float').ravel()[2:4]
        cue_pos = self.convert_jov_pos(cue_pos)
        return cue_pos

    def get_bmi_df(self, bin_index, examine_trials=True):
        '''
        Inputs:
            bin_index for aligning the ephys_time with the bmi_time (especially for aligning LFP data)
                To load unit: 
                >>> unit = UNIT(bin_len=bin_len, nbins=nbins)
                >>> unit.load_unitpacket('./fet.bin')
                >>> unit.bin_index

            We need this variable to create `ephys_time` = (bin_index + 1) * bin_len
            We need `ephys_time` to align with LPF and other ephys data. 

            examine_trials:
                if True: more columns will be added to the bmi_df
                if Flase: only basic columns will be added to the bmi_df ('x', 'y', 'ball_vel', 'vel_thres', 'ephys_time')

        Output:
            bmi_df: a dataframe with each row corresponds to a single bmi decoding output at each bin
                    with its ephys time in one of the columns

        Variables (every bmi output bin has a corresponding below variables):
            bmi_df.x: x position of the bmi output in the maze coordinate (e.g. between -50 to 50 cm)
            bmi_df.y: y position of the bmi output in the maze coordinate (e.g. between -50 to 50 cm)
            bmi_df.ball_vel: ball velocity at that bmi output bin
            bmi_df.vel_thres: ball velocity threshold preventing teleportation if animal move ball faster than the threshold
            bmi_df.jov_x: x position reported by jovian after the bmi output bin (maze coordination e.g. between -50 to 50 cm)
            bmi_df.jov_y: y position reported by jovian after the bmi output bin (maze coordination e.g. between -50 to 50 cm)
            bmi_df.jov_hd: head direction angle at the bmi output bin
            bmi_df.hd_x: head direction x component at the bmi output bin (maze coordination)
            bmi_df.hd_y: head direction y component at the bmi output bin (maze coordination)

            bmi_df.ephys_time is the **END** (instead of START) time of the bmi decoding output bin
        '''

        # create bmi_pos_df
        bmi_pos_df = self.select(func='on_decode', msg='BMI').msg.str.extractall(float_pattern).unstack().astype('float')
        bmi_pos_df.columns = ['x', 'y', 'ball_vel', 'vel_thres']

        # find the ephys_time of the bmi output bin (the time that bin ends)
        self.cell_count, self.bin_len, self.dec_len = self.bmi_params
        bmi_output_ephys_time = (bin_index + 1) * self.bin_len
        bmi_pos_df.insert(0, column='ephys_time', value=bmi_output_ephys_time)

        # find jov output in the jov_df that just before bmi output index in bmi_pos_df
        bmi_jov_df = self.jov_pos_df.iloc[self.jov_pos_df.index.searchsorted(bmi_pos_df.index)].astype('float')
        jov_pos = self.convert_jov_pos(bmi_jov_df[['jov_x', 'jov_y']].to_numpy())
        jov_pos_z = bmi_jov_df['jov_z'].to_numpy().reshape(-1, 1)
        jov_hd = bmi_jov_df['jov_hd'].to_numpy().reshape(-1, 1)
        bmi_pos_df[['jov_x', 'jov_y', 'jov_z', 'jov_hd']] = np.hstack((jov_pos, jov_pos_z, jov_hd))

        # calculate hd_xy for plotting head direction, for every bmi output index i, 
        # the animal head direction is represented by (hd_x[i], hd_y[i])
        hd = jov_hd.ravel() - 90 
        hd_xy = np.vstack((np.cos(hd/360*np.pi*2), np.sin(hd/360*np.pi*2))).T
        bmi_pos_df[['hd_x', 'hd_y']] = np.round(hd_xy, 2)

        if examine_trials: # assign trials, cues, goal_distance, and trial types to each output bin
            self.read_bmi_trials(bmi_pos_df)

        self.bmi_df = bmi_pos_df
        return bmi_pos_df

    def read_bmi_trials(self, bmi_pos_df):
        bmi_pos_df['trial'] = np.nan
        bmi_pos_df['trial_type'] = np.nan
        bmi_pos_df['cue_x'] = np.nan
        bmi_pos_df['cue_y'] = np.nan
        bmi_pos_df['goal_dist'] = np.nan

            # Extract bmi trial info 
        trial_index = self.get_trial_index()
        for trial_no in tqdm(range(0, len(trial_index))):
            trial_df = self.trial_df_orig[trial_no]
            # jov_pos, jov_hd, jov_ball_vel = self.get_jov_after_bmi(trial_no)
            trial_bmi_idx = trial_df[trial_df.func.str.contains('decode') & trial_df.msg.str.contains('BMI')].index
            cue_pos = self.get_cue_pos(trial_no)
            bmi_pos = bmi_pos_df.loc[trial_bmi_idx][['x','y']].to_numpy()

            goal_dist = np.linalg.norm(bmi_pos - cue_pos, axis=1)
            goal_radius = 22.2
            if any(goal_dist<goal_radius):
                if 1<np.where(goal_dist<goal_radius)[0][0]<=28 and len(goal_dist)<=29:
                    trial_type = 0
                elif np.where(goal_dist<goal_radius)[0][0]<=1 and len(goal_dist)<=29:
                    trial_type = -1
                    print(f'unidentified trials: {trial_no}')
                elif len(goal_dist) > 127:
                    trial_type = 2
                else:
                    trial_type = 1
            else:
                trial_type = -1
                    
            bmi_pos_df.loc[trial_bmi_idx, 'trial'] = trial_no
            bmi_pos_df.loc[trial_bmi_idx, 'trial_type'] = trial_type
            bmi_pos_df.loc[trial_bmi_idx, 'cue_x'] = cue_pos[0]
            bmi_pos_df.loc[trial_bmi_idx, 'cue_y'] = cue_pos[1]
            bmi_pos_df.loc[trial_bmi_idx, 'goal_dist'] = goal_dist

        bmi_pos_df.reset_index(inplace=True)
        bmi_pos_df.rename(columns={"index": "log_index"}, inplace=True)
        # bmi_pos_df['y'] = -bmi_pos_df['y']
        # bmi_pos_df['jov_y'] = -bmi_pos_df['jov_y']
        # bmi_pos_df['cue_y'] = -bmi_pos_df['cue_y']
        # bmi_pos_df['hd_y'] = -bmi_pos_df['hd_y']
        self.bmi_df = bmi_pos_df

    def get_bmi_trial_time(self, trial_no):
        t0, t1 = self.bmi_df[self.bmi_df.trial == trial_no].ephys_time.to_numpy()[[0, -1]]
        return t0, t1

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
        goal_pos = self.convert_jov_pos(cue_pos)

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
