import numpy as np
from spiketag.base import prb_bowtie_L as prb 
from spiketag.view import probe_view


if __name__ == '__main__':
    print(prb)
    prb_view = probe_view()
    prb_view.set_data(prb, font_size=17)
    prb_view.run()
