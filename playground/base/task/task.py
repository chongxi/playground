import numpy as np
from ..jovian import Jovian
import time


def _cue_generate_2d_maze(*args):
    '''generate a cue position that is not going to be near to the _pos in args
       low, high are border condition for cue generation
       function should return < 100us
    '''
    radius = 70

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
        @self.jov.connect
        def on_touch(args):
            self.touch_id, self.coord = args
            self.on_event(self.touch_id) 

        # fsm will use jov real-time event to update
        self.fsm = fsm 
        # reset will use jov.current_pos to generate cue
        # self.reset()



    @property
    def current_pos(self):
        '''shared memory of current pos of animal
        '''
        return self.jov.current_pos.numpy()


    def __call__(self, fsm):
        self.fsm = fsm
        self.reset()

    def on_event(self, event):
        # try:
        print self.state
        next_state, func, args = self.fsm[self.state][event]
        func(args)
        self.state = next_state
        print self.state
        # except:
        #     print('Your Finite State Machine is Incomplete or Wrong')



#------------------------------------------------------------------------------
# one cue task
#------------------------------------------------------------------------------
class one_cue_task(Task):

    def __init__(self, jov):

        fsm = {
                '1cue': {'_dcue_000': ['1cue', self.goal_cue_touched, 'reward']} 
              }

        super(one_cue_task, self).__init__(fsm, jov)

        self.cues_name = ['_dcue_000', '_dcue_001'] # goal cue: 000, guide cue: 001 (not in use)

        @self.jov.connect
        def on_animate():
            if self.z >= 0:
                self.jov.teleport(prefix='model', target_pos=[self._coord_goal[0],  self._coord_goal[1],  self.z], target_item='_dcue_000')
                self.z -= 1
            else:
                pass

        self.jov.teleport(prefix='model', target_pos=(1000, 1000, 1000), target_item='_dcue_001')
        self.z = 60
        self.reset()
    #---------------------------------------------------------------------------------------------------
    # Every task cycle finished, you need to reset (regenerate cue based on current coordination etc..)
    #---------------------------------------------------------------------------------------------------
    def reset(self):
        self._corrd_animal = self.jov._to_maze_coord(self.current_pos)[:2]
        self._coord_goal   = _cue_generate_2d_maze(self._corrd_animal) 
        self.z = 60
        self.state = '1cue'

    def goal_cue_touched(self, args):
        print(args)
        # self.jov.teleport(prefix='model', target_pos=(1000, 1000, 1000), target_item='_dcue_000')
        #TODO: give reward
        # self.jov.reward(100)
        self.reset()




#------------------------------------------------------------------------------
# one cue task
#------------------------------------------------------------------------------
class two_cue_task(Task):

    def __init__(self, jov):

        # goal cue: 000, guide cue: 001

        fsm = {
                '2cue': { '_dcue_000': ['2cue', self.warn, 'touch wrong cue'],    '_dcue_001': ['1cue', self.guide_cue_touched, 'right cue'] }, 
                '1cue': { '_dcue_000': ['2cue', self.goal_cue_touched, 'reward'], '_dcue_001': ['1cue', self.guide_cue_touched, 'wrong cue'] } 
              }

        super(two_cue_task, self).__init__(fsm, jov)

        self.cues_name = ['_dcue_000', '_dcue_001']  # goal cue: 000, guide cue: 001

        @self.jov.connect
        def on_animate():
            if self.z >= 0:
                self.jov.teleport(prefix='model', target_pos=[self._coord_guide[0], self._coord_guide[1], self.z], target_item='_dcue_001')
                self.jov.teleport(prefix='model', target_pos=[self._coord_goal[0],  self._coord_goal[1],  self.z], target_item='_dcue_000')
                self.z -= 1
            else:
                pass

        self.z = 60
        self.reset()

    #---------------------------------------------------------------------------------------------------
    # Every task cycle finished, you need to reset (regenerate cue based on current coordination etc..)
    #---------------------------------------------------------------------------------------------------
    def reset(self):
        self._corrd_animal = self.jov._to_maze_coord(self.current_pos)[:2]
        self._coord_guide  = _cue_generate_2d_maze(self._corrd_animal) 
        self._coord_goal   = _cue_generate_2d_maze(self._corrd_animal, self._coord_guide)
        self.z = 60
        # self.jov.teleport(prefix='model', target_pos=self._coord_guide, target_item='_dcue_001')
        # self.jov.teleport(prefix='model', target_pos=self._coord_goal,  target_item='_dcue_000')
        self.state = '2cue'


    def warn(self, args):
        # TODO: give sound
        print(args)

    def guide_cue_touched(self, args):
        print(args)
        self.jov.teleport(prefix='model', target_pos=(1000, 1000, 1000), target_item='_dcue_001')

    def goal_cue_touched(self, args):
        print(args)
        self.jov.teleport(prefix='model', target_pos=(1000, 1000, 1000), target_item='_dcue_000')
        #TODO: give reward
        # self.jov.reward(100)
        self.reset()





if __name__ == '__main__':
    from playground.base import Jovian
    jov = Jovian()
    task = two_cue_task('2cue', jov)
    jov.emit('touch', args=(0, (0,0,0)))


