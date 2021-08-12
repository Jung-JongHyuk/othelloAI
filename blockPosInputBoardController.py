from blockPosInputBoardView import BlockPosInputBoardView
from PyQt5.QtWidgets import QApplication

class BlockPosInputBoardViewController:
    def __init__(self, boardSize, parent):
        self.boardSize = boardSize
        self.view = BlockPosInputBoardView(self.boardSize)
        self.connectEventHandler()
    
    def connectEventHandler(self):
        pass

if __name__ == "__main__":
    app = QApplication([])
    boardViewController = BlockPosInputBoardViewController((6,6), None)
    app.exec_()