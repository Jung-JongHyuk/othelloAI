from PyQt5.QtWidgets import *
from boardWidget import BoardWidget

class BlockPosInputBoardView(QWidget):
    def __init__(self, boardSize):
        super().__init__()
        self.boardSize = boardSize
        self.boardWidget = BoardWidget(self.boardSize, self.getProperGridSize())
        self.initView()
        self.show()
    
    def initView(self):
        (rowSize, colSize) = self.boardSize
        self.setWindowTitle("Othello")
        mainLayout = QGridLayout()
        mainLayout.addWidget(QPushButton("button"), 0, 0)
        mainLayout.addWidget(self.boardWidget, 1, 0)
        self.setLayout(mainLayout)
    
    def getProperGridSize(self):
        (rowSize, colSize) = self.boardSize
        screenSize = QApplication.primaryScreen().size()
        (screenHeight, screenWidth) = (screenSize.height() * 0.5, screenSize.width() * 0.5)
        if screenHeight < screenWidth:
            return screenHeight / colSize
        else:
            return screenWidth / colSize

if __name__ == "__main__":
    app = QApplication([])
    boardViewController = BlockPosInputBoardView((6,6))
    app.exec_()

        
