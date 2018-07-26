import numpy as np
from spiketag.probe import prb_bowtie_L as prb 
from spiketag.view import probe_view

@prb.connect
def on_select(group_id, chs):
    print(group_id, chs)

if __name__ == '__main__':
    print(prb)
    # prb.prb_view.electrode_pads
    prb.show()
    # prb_view = probe_view()
    # prb_view.set_data(prb, font_size=17)
    # prb_view.run()
