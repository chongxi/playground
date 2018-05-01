import numpy as np
from ..jovian import Jovian
import time
import torch
from itertools import chain
from collections import deque


def _cue_generate_2d_maze(*args):
    '''generate a cue position that is not going to be near to the _pos in args
       low, high are border condition for cue generation
       function should return < 100us
    '''
    radius = 50 

    while True:
        break_condition = True
        pos = np.random.randint(low=-85, high=85, size=(2,))
        for _pos in args:
            if np.linalg.norm(pos - np.array(_pos)) < radius:
               break_condition = False 
        if break_condition:
            break
    return pos



class Task(object):
    '''Every task has the same jov to report event
       Every task has different fsm is defined in child task class
    ''' 
    def __init__(self, fsm, jov):
        # jovian first
        self.jov = jov
        self.animation = {}  # {"_dcue_000": deque([ (4, parachute), (60, vibrate) ]), "_dcue_111": deque([ (2, animation111) ])}

        @self.jov.connect
        def on_start():
            self.reset()  

        @self.jov.connect
        def on_touch(args):
            self.touch_id, self.coord = args
            self.on_event(self.touch_id) 

        @self.jov.connect
        def on_animate():
            for _cue_name, _action_queue in self.animation.items():
                try: 
                    _pace, _animation = _action_queue[0]
                    if self.jov.cnt % _pace == 0:
                        try:
                            _animation.next()
                        except StopIteration:
                            _action_queue.popleft()
                except:
                    pass

        # fsm will use jov real-time event to update
        self.fsm = fsm 

    @property
    def current_pos(self):
        '''shared memory of current pos of animal
        '''
        return self.jov.current_pos.numpy()

    def __call__(self, fsm):
        self.fsm = fsm
        self.reset()

    def on_event(self, event):
        try:
            print self.state
            next_state, func, args = self.fsm[self.state][event]
            func(args)
            self.state = next_state
            print self.state
        except:
            pass
        # except:
        #     print('Your Finite State Machine is Incomplete or Wrong')

    #------------------------------------------------------------------------------
    # animation effect
    #------------------------------------------------------------------------------

    def parachute(self, cue_name, pos):
        ''' usage:
            self.animation['_dcue_000'] = deque([ (3, self.parachute('_dcue_000', self._coord_goal)) ])
        '''
        for z in range(60,-1,-2):
            self.jov.teleport(prefix='model', target_pos=[pos[0],  pos[1],  z], target_item=cue_name)
            yield

    def vibrate(self, cue_name):
        ''' usage:
            self.animation['_dcue_001'] = deque([ (3, self.parachute('_dcue_001', self._coord_guide)), (30, self.vibrate('_dcue_001')) ])
        '''
        for z in range(1000):
            pos = self.jov._to_maze_coord(self.jov.shared_cue_dict[cue_name])
            self.jov.teleport(prefix='model', target_pos=[pos[0],  pos[1],  5], target_item=cue_name)
            yield
            pos = self.jov._to_maze_coord(self.jov.shared_cue_dict[cue_name])
            self.jov.teleport(prefix='model', target_pos=[pos[0],  pos[1],  0], target_item=cue_name)
            yield            

    def trajectory_teleport(self, trajectory):
        ''' usage:
            trajectory = np.arange(-99,100).repeat(2).reshape(-1,2)
            self.animation['animal'] = deque([ (3, self.trajectory_teleport(trajectory)) ])
        '''
        for pos in trajectory:
            self.jov.teleport(prefix='console', target_pos=pos)
            yield



#------------------------------------------------------------------------------
# one cue task
#------------------------------------------------------------------------------
class one_cue_task(Task):

    def __init__(self, jov):

        fsm = {
                '1cue': {'_dcue_000': ['1cue', self.goal_cue_touched, 'reward']} 
              }

        super(one_cue_task, self).__init__(fsm, jov)

        self.jov.teleport(prefix='model', target_pos=(1000, 1000, 1000), target_item='_dcue_001')
    #---------------------------------------------------------------------------------------------------
    # Every task cycle finished, you need to reset (regenerate cue based on current coordination etc..)
    #---------------------------------------------------------------------------------------------------
    def reset(self):
        self.jov.teleport(prefix='model', target_pos=(1000, 1000, 1000), target_item='_dcue_000')
        self._corrd_animal = self.jov._to_maze_coord(self.current_pos)[:2]
        self._coord_goal   = _cue_generate_2d_maze(self._corrd_animal) 
        self.animation['_dcue_000'] = deque([ (3, self.parachute('_dcue_000', self._coord_goal)), (30, self.vibrate('_dcue_000')) ])
        self.state = '1cue'

    def goal_cue_touched(self, args):
        print(args)
        self.jov.reward(self.reward_time)
        self.reset() 




#------------------------------------------------------------------------------
# two cue task
#------------------------------------------------------------------------------
class two_cue_task(Task):

    def __init__(self, jov):
        # goal cue: 000, guide cue: 001
        fsm = {
                '2cue': { '_dcue_000': ['2cue', self.warn, 'touch wrong cue'],    '_dcue_001': ['1cue', self.guide_cue_touched, 'right cue'] }, 
                '1cue': { '_dcue_000': ['2cue', self.goal_cue_touched, 'reward'], '_dcue_001': ['1cue', self.guide_cue_touched, 'wrong cue'] } 
              }
        super(two_cue_task, self).__init__(fsm, jov)

    #---------------------------------------------------------------------------------------------------
    # Every task cycle finished, you need to reset (regenerate cue based on current coordination etc..)
    #---------------------------------------------------------------------------------------------------
    def reset(self):
        self.jov.teleport(prefix='model', target_pos=(1000, 1000, 1000), target_item='_dcue_000')
        self._corrd_animal = self.jov._to_maze_coord(self.current_pos)[:2]
        self._coord_guide  = _cue_generate_2d_maze(self._corrd_animal) 
        self._coord_goal   = _cue_generate_2d_maze(self._corrd_animal, self._coord_guide)
        self.animation['_dcue_000'] = deque([ (3, self.parachute('_dcue_000', self._coord_goal)) ])
        self.animation['_dcue_001'] = deque([ (3, self.parachute('_dcue_001', self._coord_guide)), (30, self.vibrate('_dcue_001')) ])
        self.state = '2cue'

    def warn(self, args):
        # TODO: give sound
        print(args)

    def guide_cue_touched(self, args):
        print(args)
        self.jov.teleport(prefix='model', target_pos=(1000, 1000, 1000), target_item='_dcue_001')
        self.animation['_dcue_000'] = deque([ (30, self.vibrate('_dcue_000')) ]) 

    def goal_cue_touched(self, args):
        print(args)
        self.jov.teleport(prefix='model', target_pos=(1000, 1000, 1000), target_item='_dcue_000')
        self.jov.reward(self.reward_time)
        self.reset()




if __name__ == '__main__':
    from playground.base import Jovian
    jov = Jovian()
    task = two_cue_task('2cue', jov)
    jov.emit('touch', args=(0, (0,0,0)))


