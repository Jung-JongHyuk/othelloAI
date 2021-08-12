from mainView import MainView
from boardViewController import BoardViewController
from blockPosInputBoardController import BlockPosInputBoardViewController
from PyQt5.QtWidgets import QApplication

class MainViewController:
    def __init__(self):
        self.view = MainView()
        self.gameSetting = self.view.getGameSetting() # init gameSetting
        self.blockPosInputBoardViewController = None
        self.boardViewController = None
        self.connectEventHandler()
    
    def connectEventHandler(self):
        self.view.blockPosTypeComboBox.currentTextChanged.connect(self.openBlockPosInputBoardWindow)
        self.view.gameStartButton.clicked.connect(self.getGameSettingInputAndOpenBoardWindow)

    def openBlockPosInputBoardWindow(self, blockPosType):
        if blockPosType == "custom":
            self.gameSetting = self.view.getGameSetting()
            self.blockPosInputBoardViewController = BlockPosInputBoardViewController(self.gameSetting["boardSetting"]["boardSize"], self)

    def getGameSettingInputAndOpenBoardWindow(self):
        self.gameSetting = self.view.getGameSetting()
        self.boardViewController = BoardViewController(**self.gameSetting)

if __name__ == "__main__":
    app = QApplication([])
    boardViewController = MainViewController()
    app.exec_()
