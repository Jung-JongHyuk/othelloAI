from board import Board
from mainView import MainView
from boardViewController import BoardViewController
from PyQt5.QtWidgets import QApplication

class MainViewController:
    def __init__(self):
        self.view = MainView()
        self.connectEventHandler()
    
    def connectEventHandler(self):
        self.view.gameStartButton.clicked.connect(self.getInputAndOpenBoardView)

    def getInputAndOpenBoardView(self):
        kwargs = self.view.getInput()
        BoardViewController(**kwargs)

if __name__ == "__main__":
    app = QApplication([])
    boardViewController = MainViewController()
    app.exec_()
