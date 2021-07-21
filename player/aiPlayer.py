import torch
import numpy as np
from train.network.othelloNetWrapper import OthelloNetWrapper
from train.network.QNetWrapper import QNetWrapper
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
    def __init__(self, boardSize, folderName= './temp', modelName='best.pth.tar'):
        super().__init__()
        self.gameWrapper = OthelloGameWrapper(boardSize, blockPosType= 'none')
        # self.agent = OthelloNetWrapper(self.gameWrapper)
        self.agent = QNetWrapper(self.gameWrapper)
        self.agent.load_checkpoint(folder=folderName, filename=modelName)
    
    def decide(self, board):
        if len(board.getPlaceableCoordinates(self.playerIndex)) == 0:
            return None
        else:
            boardData = np.array(board.board)
            boardData = self.gameWrapper.getCanonicalForm(boardData, self.gameWrapper.convertToPlayerIndexInNumpy(self.playerIndex))
            (pi, v) = self.agent.predict(boardData)
            validMoves = self.gameWrapper.getValidMoves(boardData, self.gameWrapper.convertToPlayerIndexInNumpy(0))
            pi = pi * validMoves
            decision = np.argmax(pi)
            if validMoves[decision] == 0:
                validMoveIndexs = np.argwhere(validMoves == 1).reshape((-1,))
                decision = np.random.choice(validMoveIndexs)
            return self.gameWrapper.actionToPos(decision)




