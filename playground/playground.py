# -*- coding: utf-8 -*-

"""Main module."""
import sys
import time
from PyQt5.QtWidgets import QApplication
from gui import play_GUI
from base import create_logger

def main():
    logger = create_logger()
    app = QApplication(sys.argv)
    gui = play_GUI(logger)
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
