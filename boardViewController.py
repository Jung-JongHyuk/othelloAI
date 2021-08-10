from numpy.core.fromnumeric import searchsorted
from board import Board
from game import Game
from boardView import BoardView
from player.dummyPlayer import DummyPlayer
from player.randomPlayer import RandomPlayer
from player.alphaBetaPruningPlayer import AlphaBetaPruningPlayer
from player.AIPruningPlayer import AIPruningPlayer
from player.aiPlayer import AIPlayer
import copy

class BoardViewController:
    def __init__(self, player0, player1, boardSetting):
        self.boardSize = boardSetting["boardSize"]
        self.view = BoardView(self.boardSize)
        self.board = Board(**boardSetting)
        self.prevBoard = copy.deepcopy(self.board)
        self.Players = (DummyPlayer(), RandomPlayer())
        self.game = Game(self.board, self.Players)
        self.lastPlacedPos = []
        self.view.setPlayerName(0, f"{type(self.Players[0]).__name__} {self.view.icons[0]}")
        self.view.setPlayerName(1, f"{self.view.icons[1]} {type(self.Players[1]).__name__}")
        with open("./boardView.qss", 'r') as qss:
            self.view.setStyleSheet(qss.read())
        self.proceedGame()
        self.connectEventHandler()
        self.updateView()
    
    def connectEventHandler(self):
        (rowSize, colSize) = self.boardSize
        for row in range(rowSize):
            for col in range(colSize):
                self.view.grids[row][col].clicked.connect(lambda state, pos=(row,col): self.onGridClicked(pos))

    def onGridClicked(self, pos):
        self.game.adoptDecision(pos)
        self.prevBoard = copy.deepcopy(self.board)
        self.proceedGame()
        self.updateView()
        self.prevBoard = copy.deepcopy(self.board)
    
    def proceedGame(self):
        self.lastPlacedPos = []
        while not isinstance(self.game.getCurrPlayer(), DummyPlayer) and not self.game.isGameFinished:
            nonHumanPlayerDecision = self.game.getCurrPlayer().decide(self.board)
            self.lastPlacedPos.append(nonHumanPlayerDecision)
            print(self.game.getCurrPlayer(), nonHumanPlayerDecision)
            self.game.adoptDecision(nonHumanPlayerDecision)
    
    def updateView(self):
        (rowSize, colSize) = self.boardSize
        # update piece
        for row in range(rowSize):
            for col in range(colSize):
                self.view.setGridText((row,col), self.view.icons[self.board.board[row][col]])
                self.view.setGridClickEnabled((row,col), False)

        # update placeable pos and enable click
        placeableCoordinates = self.board.getPlaceableCoordinates(self.game.currPlayerIndex)
        for coordinate in placeableCoordinates:
            self.view.setGridText(coordinate, self.view.placeableIcons[self.game.currPlayerIndex])
            self.view.setGridClickEnabled(coordinate, True)

        # update changed and placed pos
        for row in range(rowSize):
            for col in range(colSize):
                if self.prevBoard.board[row][col] != self.board.board[row][col]:
                    self.view.setGridIsChanged((row,col), True)
                else:
                    self.view.setGridIsChanged((row,col), False)
        
        for pos in self.lastPlacedPos:
            self.view.setGridIsPlaced(pos, True)

        # update score board
        boardStatus = self.board.getBoardStatus()
        self.view.setPlayerScore(0, boardStatus[0])
        self.view.setPlayerScore(1, boardStatus[1])

        # pop alert if game is ended
        if self.game.isGameFinished:
            self.view.popAlert("Game ended.") 
