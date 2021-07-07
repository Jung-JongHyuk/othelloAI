import sys
sys.path.append('./')
from PyQt5.QtWidgets import QApplication
from boardViewController import BoardViewController

if __name__ == "__main__":
    app = QApplication([])
    boardViewController = BoardViewController((6,6), 0)
    app.exec_()