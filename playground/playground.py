# -*- coding: utf-8 -*-

"""Main module."""
import sys
import time
from PyQt5.QtWidgets import QApplication
from .gui import play_raster_GUI
from .base import create_logger, Fpga, logger
from spiketag.realtime import BMI
from spiketag.base import probe
import numpy as np
from spiketag.analysis import *
from spiketag.analysis.decoder import NaiveBayes

bin_size, B_bins = 100e-3, 6

def run():
    logger = create_logger()
    app = QApplication(sys.argv)
    bmi = BMI(fetfile='./fet.bin')
    bmi.set_binner(bin_size=bin_size, B_bins=B_bins)
    gui = play_raster_GUI(logger=logger, bmi=bmi)
    gui.show()
    sys.exit(app.exec_())


def build_decoder(bmi, spktag_file, pos_file):
    # For Lab
    log = logger(pos_file, sync=True)
    ts, pos = log.to_trajectory(session_id=0)
    pc = place_field(pos=pos, ts=ts, bin_size=2.5, v_cutoff=5)
    pc.load_spkdf(spktag_file)
    dec = NaiveBayes(t_step=bin_size, t_window=B_bins*bin_size)
    dec.connect_to(pc)
    bmi.set_decoder(dec, dec_result_file='./decoded_pos.bin')

    # For test: Using Brian's data to test system
    # pos = np.fromfile(pos_file).reshape(-1,2)
    # pc = place_field(pos=pos, t_step=33.333e-3, bin_size=4, v_cutoff=25)
    # pc.load_spkdf(spktag_file, show=True, replay_offset=2.004)
    # dec = NaiveBayes(t_step=bin_size, t_window=B_bins*bin_size)
    # dec.connect_to(pc)
    # bmi.set_decoder(dec, dec_result_file='./decoded_pos.bin')


if __name__ == '__main__':
    run()
