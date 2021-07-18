import sys
import numpy as np
import random
from .GameModel import GameModel
sys.path.append('../')
from board import Board

class OthelloGameWrapper(GameModel):
    def __init__(self, boardSize, blockPosType= 0):
        self.boardSize = boardSize
        self.blockPosType = blockPosType
        if self.boardSize[0] != self.boardSize[1] and self.blockPosType != 0:
            print("invalid blockPosType")
            exit()

    def getInitBoard(self):
        board = Board(self.boardSize, numOfBlock= 0)
        if self.blockPosType == 1: # type 1: cross shape
            board.setBlock([(0,0), (1,1), (2,2), (0,7), (1,6), (2,5), (7,0), (6,1), (5,2), (7,7), (6,6), (5,5)])
        elif self.blockPosType == 2: # type 2: center divided
            board.setBlock([(0,3), (0,4), (1,3), (1,4), (3,0), (4,0), (3,1), (4,1), (7,3), (7,4), (6,3), (6,4), (3,7), (3,6), (4,7), (4,6)])
        return np.array(board.board)
    
    def getBoardSize(self):
        return self.boardSize
    
    def getActionSize(self):
        return self.boardSize[0] * self.boardSize[1] + 1
    
    def getNextState(self, board, player, action):
        if action == self.boardSize[0] * self.boardSize[1]:
            return (board, -player)
        else:
            currBoard = self.convertToBoardClass(board)
            posToPlace = self.actionToPos(action)
            currBoard.placePiece(posToPlace, self.convertToPlayerIndexInBoardClass(player))
            return (np.array(currBoard.board), -player)
    
    def getValidMoves(self, board, player):
        valids = [0] * self.getActionSize()
        currBoard = self.convertToBoardClass(board)
        placeablePos = currBoard.getPlaceableCoordinates(self.convertToPlayerIndexInBoardClass(player))
        if len(placeablePos) == 0:
            valids[-1] = 1
        else:
            for pos in placeablePos:
                valids[self.posToAction(pos)] = 1
        return np.array(valids)
    
    def getGameEnded(self, board, player):
        currBoard = self.convertToBoardClass(board)
        if len(currBoard.getPlaceableCoordinates(self.convertToPlayerIndexInBoardClass(player))) != 0 or len(currBoard.getPlaceableCoordinates(self.convertToPlayerIndexInBoardClass(-player))) != 0:
            return 0
        else:
            boardStatus = currBoard.getBoardStatus()
            if boardStatus[self.convertToPlayerIndexInBoardClass(player)] > boardStatus[self.convertToPlayerIndexInBoardClass(-player)]:
                return 1
            elif boardStatus[self.convertToPlayerIndexInBoardClass(player)] < boardStatus[self.convertToPlayerIndexInBoardClass(-player)]:
                return -1
            else:
                return 0.1

    def getCanonicalForm(self, board, player):
        if player == 1:
            return board.copy()
        else:
            result = board.copy()
            result = np.where(result == 0, None, result)
            result = np.where(result == 1, 0, result)
            result = np.where(result == None, 1, result)
            return result
    
    def getSymmetries(self, board, pi):
        piOfBoard = np.reshape(pi[:-1], self.boardSize)
        result = []
        for i in range(1, 5):
            for j in [True, False]:
                newBoard = np.rot90(board, i)
                newPiOfBoard = np.rot90(piOfBoard, i)
                if j:
                    newBoard = np.fliplr(newBoard)
                    newPiOfBoard = np.fliplr(newPiOfBoard)
                result += [(newBoard, list(newPiOfBoard.ravel()) + [pi[-1]])]
        return result

    def stringRepresentation(self, board):
        return board.tostring()

    def convertToBoardClass(self, board):
        result = Board(self.boardSize, numOfBlock= 0)
        result.board = board.tolist()
        return result

    def actionToPos(self, action):
        if action == self.boardSize[0] * self.boardSize[1]:
            return None
        else:
            colSize = self.boardSize[1]
            return (int(action / colSize), action % colSize)
    
    def posToAction(self, pos):
        return pos[0] * self.boardSize[1] + pos[1]
    
    def convertToPlayerIndexInBoardClass(self, player):
        #1이 학습 모델의 인덱스이자 선공
        #(-1, 1) <=> (1, 0)
        return 0 if player == 1 else 1
    
    def convertToPlayerIndexInNumpy(self, player):
        #(0, 1) <=> (1, -1)
        return -1 if player == 1 else 1

