from board import Board
from game import Game
from boardView import BoardView
from player.humanPlayer import HumanPlayer
from player.randomPlayer import RandomPlayer

class BoardViewController:
    def __init__(self, boardSize, numOfBlank):
        self.boardSize = boardSize
        self.view = BoardView(boardSize)
        self.board = Board(boardSize, numOfBlank)
        self.Players = (HumanPlayer(), RandomPlayer())
        self.game = Game(self.board, self.Players)
        self.makeBoardView()
        self.updateView()
    
    def makeBoardView(self):
        (rowSize, colSize) = self.boardSize
        for row in range(rowSize):
            for col in range(colSize):
                self.view.grids[row][col].clicked.connect(lambda state, pos=(row,col): self.onGridClicked(pos))

    def onGridClicked(self, pos):
        self.game.adoptDecision(pos)
        while not isinstance(self.game.getCurrPlayer(), HumanPlayer):
            counterPlayerDecision = self.game.getCurrPlayer().decide(self.board)
            print(self.game.getCurrPlayer(), counterPlayerDecision)
            self.game.adoptDecision(counterPlayerDecision)
        self.updateView()
    
    def updateView(self):
        (rowSize, colSize) = self.boardSize
        for row in range(rowSize):
            for col in range(colSize):
                self.view.setGridText((row,col), self.view.icons[self.board.board[row][col]])
                self.view.setGridClickEnabled((row,col), False)

        placeableCoordinates = self.board.getPlaceableCoordinates(self.game.currPlayerIndex)
        for coordinate in placeableCoordinates:
            self.view.setGridText(coordinate, self.view.placeableIcons[self.game.currPlayerIndex])
            self.view.setGridClickEnabled(coordinate, True)
