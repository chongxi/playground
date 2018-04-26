        # self.tasks = {
        #               'two_cue': self.two_cue
        #               'one_cue': self.one_cue
        #               }
        # self.task  = task_name
import numpy as np

is_close = lambda p0,p1: np.allclose(p0,p1)

fsm_two_cue = {
                'S0': ('S0', ~is_close(pos, pos_cue0), None),
                'S0': ('S1',  is_close(pos, pos_cue0), jov.cue_touch()), 
                'S1': ('S2', condition[1], command[1]), 
                'S2': ('S0', condition[2], command[2])
              }



class two_cue(object):
    """docstring for State"""
    def __init__(self):
        super(two_cue, self).__init__()
        # _state is a state generator
        self.fsm = fsm
        self.state = self._state_gen()
        self._current_state = 0
        self._next_state = 1


    def _state_gen(self):
        if state_transition_condition:
            print(state_transtion)
            self._current_state = self._next_state
            # self._next_state = self.fsm[]
            yield self.fsm[self._current_state] 

    
    def next(self):
        self.state.next()



class Task(EventEmitter):
    """docstring for task"""
    def __init__(self, taskname):
        super(Task, self).__init__()
        if taskname == '2cue':
            self.fsm = two_cue()

    def connect(self, jov):
        self.jov = jov
        @self.jov.connect
        def on_touch(args):
            self.touch_id, self.coord = args
            self.fsm.next(self.touch_id)

