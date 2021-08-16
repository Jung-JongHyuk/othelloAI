from blockPosInputBoardView import BlockPosInputBoardView
from PyQt5.QtWidgets import QApplication

class BlockPosInputBoardViewController:
    def __init__(self, boardSize, parentViewController):
        self.parentViewController = parentViewController
        self.boardSize = boardSize
        self.view = BlockPosInputBoardView(self.boardSize)
        (rowSize, colSize) = self.boardSize
        self.isBlockPlaced = [[False for col in range(colSize)] for row in range(rowSize)]
        self.connectEventHandler()
        self.initView()
    
    def connectEventHandler(self):
        (rowSize, colSize) = self.boardSize
        for row in range(rowSize):
            for col in range(colSize):
                self.view.setGridClickedEventHandler((row,col), lambda state, pos=(row,col): self.onGridClicked(pos))
        self.view.submitButton.clicked.connect(self.onSubmitButtonClicked)
    
    def onGridClicked(self, pos):
        (row, col) = pos
        if self.isBlockPlaced[row][col]:
            self.isBlockPlaced[row][col] = False
            self.view.setGridText((row,col), "")
        else:
            self.isBlockPlaced[row][col] = True
            self.view.setGridText((row,col), "🪨")
    
    def onSubmitButtonClicked(self):
        self.view.accept()
    
    def getBlockPos(self):
        (rowSize, colSize) = self.boardSize
        blockPos = []
        for row in range(rowSize):
            for col in range(colSize):
                if self.isBlockPlaced[row][col]:
                    blockPos.append((row,col))
        return blockPos

    def initView(self):
        (rowSize, colSize) = self.boardSize
        for row in range(rowSize):
            for col in range(colSize):
                if not self.isBlockPlaceable((row,col)):
                    self.view.setGridIsUnplaceable((row,col), True)
                    self.view.setGridClickEnabled((row,col), False)

    def isBlockPlaceable(self, pos):
        (rowSize, colSize) = self.boardSize
        (row, col) = pos
        # check is initial
        if row >= rowSize / 2 - 1 and row <= rowSize / 2 and col >= colSize / 2 - 1 and col <= colSize / 2:
            return False
        elif row >= rowSize / 2 - 1 and row <= rowSize / 2 and (col == colSize / 2 - 2 or col == colSize / 2 + 1):
            return False
        elif col >= colSize / 2 - 1 and col <= colSize / 2 and (row == rowSize / 2 - 2 or row == rowSize / 2 + 1):
            return False
        else:
            return True


if __name__ == "__main__":
    app = QApplication([])
    boardViewController = BlockPosInputBoardViewController((6,6), None)
    app.exec_()