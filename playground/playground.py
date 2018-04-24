# -*- coding: utf-8 -*-

"""Main module."""
import sys
import time
from PyQt5.QtWidgets import QApplication
from gui import play_GUI

def main():
    # print 'playground start'
    app = QApplication(sys.argv)
    gui = play_GUI(0)
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()