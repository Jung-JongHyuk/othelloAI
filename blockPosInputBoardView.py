from PyQt5.QtWidgets import *
from boardWidget import BoardWidget

class BlockPosInputBoardView(QDialog):
    def __init__(self, boardSize):
        super().__init__()
        self.boardSize = boardSize
        self.boardWidget = BoardWidget(self.boardSize, self.getProperGridSize())
        self.submitButton = QPushButton("submit")
        self.initView()
        self.show()
    
    def initView(self):
        self.setWindowTitle("Othello")
        mainLayout = QGridLayout()
        mainLayout.addWidget(self.submitButton, 0, 0)
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
    
    def setGridClickedEventHandler(self, pos, handler):
        (row, col) = pos
        self.boardWidget.grids[row][col].clicked.connect(handler)
    
    def setGridIsUnplaceable(self, pos, isPlaceable):
        self.boardWidget.setGridIsUnplaceable(pos, isPlaceable)
    
    def setGridClickEnabled(self, pos, clickEnabled):
        self.boardWidget.setGridClickEnabled(pos, clickEnabled)
    
    def setGridText(self, pos, text):
        self.boardWidget.setGridText(pos, text)

if __name__ == "__main__":
    app = QApplication([])
    boardViewController = BlockPosInputBoardView((6,6))
    app.exec_()

        
