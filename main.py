from PyQt5.QtWidgets import QApplication
from boardViewController import BoardViewController

if __name__ == "__main__":
    app = QApplication([])
    boardViewController = BoardViewController((8,8), 5)
    app.exec_()