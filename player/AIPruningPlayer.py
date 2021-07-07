from game import Game
from player.alphaBetaPruningPlayer import AlphaBetaPruningPlayer
from train.othelloNetWrapper import OthelloNetWrapper
from train.utils import *
from train.othelloGameWrapper import OthelloGameWrapper
import numpy as np
import torch

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})

class AIPruningPlayer(AlphaBetaPruningPlayer):
    def __init__(self, boardSize, modelName, seachDepth):
        super().__init__(seachDepth)
        self.gameWrapper = OthelloGameWrapper(boardSize[0])
        self.agent = OthelloNetWrapper(self.gameWrapper)
        self.agent.load_checkpoint(folder='./temp', filename=modelName)
    
    def getScore(self, currGame):
        boardData = np.array(currGame.board.board)
        boardData = self.gameWrapper.getCanonicalForm(boardData, self.gameWrapper.convertToPlayerIndexInNumpy(self.playerIndex))
        (pi, v) = self.agent.predict(boardData)
        return v
