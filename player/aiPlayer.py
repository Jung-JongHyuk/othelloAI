import torch
import numpy as np
from train.othelloNetWrapper import OthelloNetWrapper
from train.utils import *
from train.othelloGameWrapper import OthelloGameWrapper
from player.playerInterface import PlayerInterface

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})

class AIPlayer(PlayerInterface):
    def __init__(self, boardSize, modelName='best.pth.tar'):
        super().__init__()
        self.gameWrapper = OthelloGameWrapper(boardSize)
        self.agent = OthelloNetWrapper(self.gameWrapper)
        self.agent.load_checkpoint(folder='./temp', filename=modelName)
    
    def decide(self, board):
        boardData = np.array(board.board)
        boardData = self.gameWrapper.getCanonicalForm(boardData, self.gameWrapper.convertToPlayerIndexInNumpy(self.playerIndex))
        (pi, v) = self.agent.predict(boardData)
        pi = pi * self.gameWrapper.getValidMoves(boardData, self.gameWrapper.convertToPlayerIndexInNumpy(0))
        decision = np.argmax(pi)
        return self.gameWrapper.actionToPos(decision)




