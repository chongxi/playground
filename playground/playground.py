# -*- coding: utf-8 -*-

"""Main module."""
import sys
import time
from PyQt5.QtWidgets import QApplication
from .gui import play_GUI
from .base import create_logger, Fpga
from spiketag.realtime import BMI
from spiketag.base import probe

# def run(fpga=False):
#     logger = create_logger()
#     app = QApplication(sys.argv)
#     if fpga:
#         fpga   = Fpga(prb)
#         gui = play_GUI(logger, prb, fpga)
#     else:
#         gui = play_GUI(logger, prb)
#     gui.show()
#     sys.exit(app.exec_())

def run(prb_file, BMI_ON=False):
    logger = create_logger()
    app = QApplication(sys.argv)
    prb = probe(prb_file)
    if BMI_ON:
        bmi = BMI(prb, './fet.bin')
        gui = play_GUI(logger, prb, bmi)
    else:
        gui = play_GUI(logger, prb)
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run()
