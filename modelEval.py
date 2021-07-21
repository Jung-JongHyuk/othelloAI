from player.aiPlayer import AIPlayer
from train.ELO import *
from train.network.QNetWrapper import QNetWrapper
from train.othelloGameWrapper import OthelloGameWrapper
from trainAI import args
from train.MCTS import MCTS
from train.Arena import Arena
from game import Game
from board import Board
import os
import numpy as np

class Evaluator:
    def __init__(self, boardSize, blockPosType, folderName):
        self.models = []
        self.boardSize = boardSize
        self.blockPosType = blockPosType
        self.elo = ELO()
        self.numOfModel = len(os.listdir(folderName))

    def evalModels(self):
        self.elo.addPlayer(str(0))
        for i in range(1, self.numOfModel):
            self.elo.addPlayer(str(i), rating= self.elo.getPlayerRating(str(i - 1)))
            for j in range(0, i):
                winner = self.playSingleGame((i, j))
                print(f"before: {i}: {self.elo.getPlayerRating(str(i))}, {j}: {self.elo.getPlayerRating(str(j))}")
                print(f"{(i,j)} -> winner : {winner}")
                if winner != None:
                    self.elo.recordMatch(str(i), str(j), winner= str(winner))
                else:
                    self.elo.recordMatch(str(i), str(j), draw= True)
                print(f"after: {i}: {self.elo.getPlayerRating(str(i))}, {j}: {self.elo.getPlayerRating(str(j))}")
                print(self.elo.getRatingList())
            
            for j in range(0, i):
                winner = self.playSingleGame((j, i))
                print(f"before: {i}: {self.elo.getPlayerRating(str(i))}, {j}: {self.elo.getPlayerRating(str(j))}")
                print(f"{(i,j)} -> winner : {winner}")
                if winner != None:
                    self.elo.recordMatch(str(i), str(j), winner= str(winner))
                else:
                    self.elo.recordMatch(str(i), str(j), draw= True)
                print(f"after: {i}: {self.elo.getPlayerRating(str(i))}, {j}: {self.elo.getPlayerRating(str(j))}")
                print(self.elo.getRatingList())
            # print(self.elo.getRatingList())

    def playSingleGame(self, modelIdx):
        prefix = "QNetWrapper_(6, 6)_none_checkpoint_"

        # game = OthelloGameWrapper(self.boardSize, self.blockPosType)
        # (model0, model1) = (QNetWrapper(game), QNetWrapper(game))
        # model0.load_checkpoint(folder= "./model", filename= prefix + str(modelIdx[0] + 1) + ".pth.tar")
        # model1.load_checkpoint(folder= "./model", filename= prefix + str(modelIdx[1] + 1) + ".pth.tar")
        # tree0 = MCTS(game, model0, args)
        # tree1 = MCTS(game, model1, args)
        # arena = Arena(lambda x: np.argmax(tree0.getActionProb(x, temp=0)), lambda x: np.argmax(tree1.getActionProb(x, temp=0)), game)
        # result = arena.playGame()
        # if result == 1:
        #     return 0
        # elif result == -1:
        #     return 1
        # else:
        #     return None

        player0 = AIPlayer(self.boardSize, folderName= "./model", modelName= prefix + str(modelIdx[0] + 1) + ".pth.tar")
        player1 = AIPlayer(self.boardSize, folderName= "./model", modelName= prefix + str(modelIdx[1] + 1) + ".pth.tar")
        game = Game(Board(self.boardSize, mode= "none", blockPosType= self.blockPosType), (player0, player1))
        print(prefix + str(modelIdx[0] + 1) + ".pth.tar")
        print(prefix + str(modelIdx[1] + 1) + ".pth.tar")
        winner = game.play()
        print(winner)

        if winner == 0:
            return modelIdx[0]
        elif winner == 1:
            return modelIdx[1]
        else:
            return None

eval = Evaluator(boardSize= (6,6), blockPosType= "none", folderName= "./model")
print(eval.playSingleGame((27,14)))
# eval.evalModels()

boardSize = (6,6)

player0 = AIPlayer(boardSize, folderName= './model', modelName='QNetWrapper_(6, 6)_none_checkpoint_28.pth.tar')
# player1 = AIPruningPlayer((6,6), modelName='best.pth.tar', seachDepth=3)
player1 = AIPlayer(boardSize, folderName= './model', modelName='QNetWrapper_(6, 6)_none_checkpoint_15.pth.tar')
# player1 = RandomPlayer()

wins = [0,0]
for i in range(10):
    board = Board(boardSize)
    game = Game(board, (player0, player1))
    game = Game(Board(boardSize, mode= "none", blockPosType= "none"), (player0, player1))
    winner = game.play()
    # winner = game.play(printBoard= False)
    print(winner)
    if winner != None:
        wins[winner] = wins[winner] + 1
    print(wins)