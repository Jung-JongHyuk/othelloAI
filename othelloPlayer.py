import sys
sys.path.append('./')
from PyQt5.QtWidgets import QApplication
from boardViewController import BoardViewController

if __name__ == "__main__":
    app = QApplication([])
    boardViewController = BoardViewController({"boardSize":(8,8), "numOfBlock":0})
    app.exec_()