# -*- coding: utf-8 -*-

"""Main module."""
import sys
import time
from PyQt5.QtWidgets import QApplication
from .gui import NAV_GUI
from spiketag.realtime import BMI
import numpy as np
import torch
import pandas as pd
from spiketag.analysis import *
from spiketag.analysis.decoder import NaiveBayes

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

def run_experiment(bmi_update_rule, posterior_threshold, bmi_mode):
    app = QApplication(sys.argv)
    if bmi_mode:
        bmi = BMI(fetfile='./fet.bin')
        bmi.bmi_update_rule = bmi_update_rule
        bmi.posterior_threshold = posterior_threshold
        bmi.two_steps = two_steps_decoding
    else:
        bmi = None
    gui = NAV_GUI(bmi=bmi)
    gui.show()
    sys.exit(app.exec_())

def build_decoder(bmi, dec_file):
    # For Lab
    # log = logger(pos_file)  # pos_file is process.log
    # pc = log.to_pc(bin_size=4, v_cutoff=4)
    # # pc.align_with_recording(0, ephys_end_time)  # check if this is necessary in lab test
    # # pc.initialize()
    # pc.load_spkdf(spktag_file)
    # # check and store the cross-validation score
    # _, score = pc.to_dec(t_step=bin_size, t_window=bin_size*B_bins, t_smooth=t_smooth, 
    #                        first_unit_is_noise=True,  min_bit=0.1, min_peak_rate=0.8, 
    #                        firing_rate_modulation=True, verbose=True,
    #                        training_range = [0.00, 0.60],
    #                        testing_range  = [0.60, 1.00],
    #                        low_speed_cutoff = {'training': True, 'testing': True})

    # # use the full recorded data (meaning no cross-validaton) for BMI 
    # dec, _ = pc.to_dec(t_step=bin_size, t_window=bin_size*B_bins, t_smooth=t_smooth, 
    #                        first_unit_is_noise=True,  min_bit=0.1, min_peak_rate=0.8, 
    #                        firing_rate_modulation=True, verbose=True,
    #                        training_range = [0.00, 1.00],
    #                        testing_range  = [0.00, 1.00],
    #                        low_speed_cutoff = {'training': True, 'testing': True})
    dec = torch.load(dec_file)
    bmi.set_binner(bin_size=dec.t_step, B_bins=dec.B_bins) # ! set binner to get real time scv correctly
    bmi.set_decoder(dec)
    bmi.pos_buffer_len = dec.smooth_factor # ! position buffer length for moving average
    bmi.mean_firing_rate = np.mean(dec.train_X[:, dec.neuron_idx])  # average firing rate of all cells over all time bins
    
    # @bmi.binner.connect
    # def on_decode(X):
    #     print(X.shape)
    #     print(type(X))
    #     print(X.dtype)
    #     f_scv = open('./scv.bin', 'ab+')
    #     f_scv.write(X.tobytes())
    #     f_scv.close()
    #     y = bmi.dec.model.predict_rt(
    #         X, bmi.dec.neuron_idx, cuda=False, mode='eval', bn_momentum=0.9)
    #     print(y)
    
    return dec._score

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
    run_experiment()
