import sys
sys.path.append('./')
from PyQt5.QtWidgets import QApplication
from boardViewController import BoardViewController

if __name__ == "__main__":
    app = QApplication([])
    boardViewController = BoardViewController((8,8), 0)
    app.exec_()