import sys
import time
from PyQt5.QtWidgets import QApplication
from playground.gui import play_GUI
from playground.base import create_logger
from spiketag.probe import prb_bowtie_LL as prb 

if __name__ == '__main__':
    logger = create_logger()
    app = QApplication(sys.argv)
    gui = play_GUI(logger, prb)
    gui.show()
    sys.exit(app.exec_())
