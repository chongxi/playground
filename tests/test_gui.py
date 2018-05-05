import sys
import time
from PyQt5.QtWidgets import QApplication
from playground.gui import play_GUI
from playground.base import create_logger

if __name__ == '__main__':
    logger = create_logger()
    app = QApplication(sys.argv)
    gui = play_GUI(logger)
    gui.show()
    sys.exit(app.exec_())
