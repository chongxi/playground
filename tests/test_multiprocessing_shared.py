from torch import multiprocessing
from playground.utils import Timer

def worker(d, key, value):
    if key%2==0:
        d[key] = value
    else:
        d[key] = None

if __name__ == '__main__':
    mgr = multiprocessing.Manager()
    d = mgr.dict()
    jobs = [ 
             multiprocessing.Process(target=worker, args=(d, i, {'name':'cue'+str(i+1),
                                                                 'pos' :(i*2, i**2, (i+1)**2),
                                                                 'changed': 1}))
             for i in range(10) 
           ]
    with Timer('', verbose=True):
        [j.start() for j in jobs]
        [j.join() for j in jobs]
    for i in d.items():
        print i 
