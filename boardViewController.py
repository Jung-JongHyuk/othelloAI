from board import Board
from game import Game
from boardView import BoardView

class BoardViewController:
    def __init__(self, boardSize, numOfBlank):
        self.board = Board(boardSize, numOfBlank)
        self.view = BoardView(boardSize)
        self.boardSize = boardSize
        self.makeBoardView()
        self.updateView()
    
    def makeBoardView(self):
        (rowSize, colSize) = self.boardSize
        for row in range(rowSize):
            for col in range(colSize):
                self.view.grids[row][col].clicked.connect(lambda state, pos=(row,col): self.onGridClicked(pos))

    def onGridClicked(self, pos):
        self.board.placePiece(pos, self.board.currentTurnPlayer)
        self.updateView()
    
    def updateView(self):
        (rowSize, colSize) = self.boardSize
        for row in range(rowSize):
            for col in range(colSize):
                self.view.setGridText((row,col), self.view.icons[self.board.board[row][col]])
                self.view.setGridClickEnabled((row,col), False)

        placeableCoordinates = self.board.getPlaceableCoordinates(self.board.currentTurnPlayer)
        for coordinate in placeableCoordinates:
            self.view.setGridText(coordinate, self.view.placeableIcons[self.board.currentTurnPlayer])
            self.view.setGridClickEnabled(coordinate, True)
