import sys
import time
from PyQt5.QtWidgets import QApplication
from playground.gui import play_GUI


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = play_GUI()
    gui.show()
    sys.exit(app.exec_())
