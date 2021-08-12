from PyQt5.QtWidgets import *

class BlockPosInputBoardView(QWidget):
    def __init__(self, boardSize):
        super().__init__()
        self.boardSize = boardSize
        self.initView()
        self.show()
    
    def initView(self):
        (rowSize, colSize) = self.boardSize
        self.setWindowTitle("Othello")
        self.grids = [[QPushButton('', self) for col in range(colSize)] for row in range(rowSize)]
        gridLayout = QGridLayout()
        gridSize = self.getProperGridSize()
        for row in range(rowSize):
            for col in range(colSize):
                self.grids[row][col].setFixedWidth(gridSize)
                self.grids[row][col].setFixedHeight(gridSize)
                gridLayout.addWidget(self.grids[row][col], row + 1, col)
        self.setLayout(gridLayout)
    
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

        
