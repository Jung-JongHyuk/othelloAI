from player.playerInterface import PlayerInterface
from player.dummyPlayer import DummyPlayer
from game import Game
import sys
import copy

class AlphaBetaPruningPlayer(PlayerInterface):
    def __init__(self, seachDepth):
        super().__init__()
        self.searchDepth = seachDepth

    def decide(self, board):
        game = Game(board, (DummyPlayer(), DummyPlayer()))
        game.currPlayerIndex = self.playerIndex
        placeablePos = game.board.getPlaceableCoordinates(self.playerIndex)
        bestPos = None
        bestAlpha = -(sys.maxsize + 1)
        for pos in placeablePos:
            nextGame = copy.deepcopy(game)
            nextGame.adoptDecision(pos)
            currAlpha = self.alphaBetaSearch(nextGame, -(sys.maxsize + 1), sys.maxsize, self.searchDepth)
            if currAlpha > bestAlpha:
                bestPos = pos
                bestAlpha = currAlpha
        return bestPos
    
    def alphaBetaSearch(self, currGame, alpha, beta, remainDepth):
        if remainDepth == 0 or currGame.isGameFinished:
            return self.getScore(currGame)
        elif currGame.currPlayerIndex == self.playerIndex:
            placeablePos = currGame.board.getPlaceableCoordinates(currGame.currPlayerIndex)
            if len(placeablePos) == 0:
                exit()
            for pos in placeablePos:
                nextGame = copy.deepcopy(currGame)
                nextGame.adoptDecision(pos)
                alpha = max(alpha, self.alphaBetaSearch(nextGame, alpha, beta, remainDepth - 1))
                if beta <= alpha:
                    break
            return alpha
        else:
            placeablePos = currGame.board.getPlaceableCoordinates(currGame.currPlayerIndex)
            if len(placeablePos) == 0:
                exit()
            for pos in placeablePos:
                nextGame = copy.deepcopy(currGame)
                nextGame.adoptDecision(pos)
                beta = min(beta, self.alphaBetaSearch(nextGame, alpha, beta, remainDepth - 1))
                if beta <= alpha:
                    break
            return beta

    def getScore(self, currGame):
        boardStatus = currGame.board.getBoardStatus()
        return boardStatus[self.playerIndex] - boardStatus[currGame.getCounterPlayerIndex(self.playerIndex)]