# -*- coding: utf-8 -*-

"""Main module."""
import sys
import time
from PyQt5.QtWidgets import QApplication
from .gui import play_GUI, play_raster_GUI
from .base import create_logger, Fpga
from spiketag.realtime import BMI
from spiketag.base import probe
import numpy as np
from spiketag.analysis import *
from spiketag.analysis.decoder import NaiveBayes

bin_size, B_bins = 50e-3, 5

def run(gui_type, prb_file, BMI_ON=False, DEC_ON=False):
    logger = create_logger()
    app = QApplication(sys.argv)
    prb = probe(prb_file)
    if BMI_ON:
        bmi = BMI(prb, './fet.bin')
        bmi.set_binner(bin_size=bin_size, B_bins=B_bins)
        if gui_type == 'raster':
            gui = play_raster_GUI(logger, prb, bmi)
        else:
            gui = play_GUI(logger, prb, bmi)
    else:
        gui = play_GUI(logger, prb)

    if DEC_ON:
        pos_file = '../sorting/dusty_pos.bin'
        realtime_decode(bmi, pos_file)
    gui.show()
    sys.exit(app.exec_())

def realtime_decode(bmi, pos_file):
    pos = np.fromfile(pos_file).reshape(-1,2)
    pc = place_field(pos=pos, t_step=33.333e-3)
    replay_offset = 2.004
    start = 320
    end   = 2500
    pc.align_with_recording(start, end, replay_offset)
    pc.initialize(bin_size=4, v_cutoff=25)
    pc.load_spktag('../sorting/spktag/test_allspikes', show=True)
    dec = NaiveBayes(t_step=bin_size, t_window=B_bins*bin_size)
    dec.connect_to(pc)
    bmi.set_decoder(dec, dec_result_file='./decoded_pos.bin')


if __name__ == '__main__':
    run()
