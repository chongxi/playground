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
import pandas as pd
from spiketag.analysis import *
from spiketag.analysis.decoder import NaiveBayes

bin_size, B_bins, t_smooth = 100e-3, 8, 3
pos_buffer_len = int(t_smooth / bin_size)
two_steps_decoding = False

def normalize_pos(pos, scale):
    pos = (pos - np.mean(pos,axis=0))
    # pos = pos/np.max(np.abs(pos),axis=0) * scale
    range_x = pos[:,0].max() - pos[:,0].min()
    range_y = pos[:,1].max() - pos[:,1].min()
    pos[:,0] = (pos[:,0] - pos[:,0].min() ) / range_x * scale - scale/2
    pos[:,1] = (pos[:,1] - pos[:,1].min() ) / range_x * scale - scale/2
    # pos = (pos - np.mean(pos,axis=0))
    return pos

def run(bmi_update_rule, posterior_threshold, bmi_mode):
    playground_log = create_logger()
    app = QApplication(sys.argv)
    if bmi_mode:
        bmi = BMI(fetfile='./fet.bin')
        bmi.bmi_update_rule = bmi_update_rule
        bmi.posterior_threshold = posterior_threshold
        bmi.pos_buffer_len = pos_buffer_len # position buffer length for moving average
        bmi.two_steps = two_steps_decoding
        bmi.set_binner(bin_size=bin_size, B_bins=B_bins)
    else:
        bmi = None
    gui = play_raster_GUI(logger=playground_log, bmi=bmi)
    gui.show()
    sys.exit(app.exec_())

def build_decoder(bmi, spktag_file, pos_file):
    # For Lab
    log = logger(pos_file)  # pos_file is process.log
    pc = log.to_pc(bin_size=4, v_cutoff=2)
    # pc.align_with_recording(0, ephys_end_time)  # check if this is necessary in lab test
    # pc.initialize()
    pc.load_spkdf(spktag_file)
    # dec, score = pc.to_dec(t_step=bin_size, t_window=bin_size*B_bins, t_smooth=1)  # previous versions
    dec, score = pc.to_dec(t_step=bin_size, t_window=bin_size*B_bins, t_smooth=t_smooth, 
                           first_unit_is_noise=True, peak_rate=0.8, 
                           training_range = [0.00, 0.50],
                           testing_range  = [0.50, 1.00])
    bmi.set_decoder(dec, dec_file='dec')
    bmi.mean_firing_rate = np.mean(dec.train_X[:, dec.neuron_idx])  # average firing rate of all cells over all time bins
    return score

    # For test: Using Brian's data to test system
    # dusty_pos = pd.read_pickle(pos_file)
    # t = dusty_pos.time.to_numpy()
    # x,y = dusty_pos.x.to_numpy(), dusty_pos.y.to_numpy()
    # pos = np.vstack((x,y)).T
    # pos = normalize_pos(pos, scale=100)
    # pc = place_field(pos=pos.copy(), ts=t.copy(), bin_size=2.5, v_cutoff=22)
    # pc.load_spkdf(spktag_file, show=False, replay_offset=2.38)
    # pc.align_with_recording(300, 750)  # training data are within range of [300, 750] seconds
    # dec, score = pc.to_dec(t_step=bin_size, t_window=bin_size*B_bins, t_smooth=1)
    # drop_neurons = np.where((dec.pc.metric['spatial_bit_spike']<0.1) &
    #                         (dec.pc.metric['peak_rate']>0.1))[0]
    # dec.drop_neuron(np.append(0,drop_neurons))
    # bmi.set_decoder(dec, dec_file='dec')

if __name__ == '__main__':
    run()
