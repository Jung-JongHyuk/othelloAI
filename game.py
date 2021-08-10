from random import shuffle
from train.othelloGameWrapper import OthelloGameWrapper
from train.network.PQNetWrapper import PQNetWrapper
from board import Board

class Game:
    def __init__(self, board, players):
        players[0].setPlayerIndex(0)
        players[1].setPlayerIndex(1)
        self.players = [players[0], players[1]]
        self.board = board
        self.currPlayerIndex = self.board.PLAYER_0
        self.isPrevPlayerSkipped = False
        self.isGameFinished = False
    
    # simulate entire game and return winner. If draw, return None.
    def play(self, printBoard= False):
        boardStatus = self.board.getBoardStatus()
        self.isPrevPlayerSkipped = False
        self.isGameFinished = False
        while not self.isGameFinished:
            posToPlace = self.getCurrPlayer().decide(self.board)
            self.adoptDecision(posToPlace)
            boardStatus = self.board.getBoardStatus()
            if printBoard:
                self.board.printBoard()

        if boardStatus[self.board.PLAYER_0] > boardStatus[self.board.PLAYER_1]:
            return self.board.PLAYER_0
        elif boardStatus[self.board.PLAYER_0] < boardStatus[self.board.PLAYER_1]:
            return self.board.PLAYER_1
        else:
            return None
    
    def adoptDecision(self, posToPlace):
        if posToPlace == None:
            if self.isPrevPlayerSkipped:
                self.isGameFinished = True
            self.isPrevPlayerSkipped = True
        elif posToPlace in self.board.getPlaceableCoordinates(self.currPlayerIndex):
            self.board.placePiece(posToPlace, self.currPlayerIndex)
            self.isPrevPlayerSkipped = False
            self.currPlayerIndex = self.getCounterPlayerIndex(self.currPlayerIndex)
            if len(self.board.getPlaceableCoordinates(self.currPlayerIndex)) == 0:
                self.isPrevPlayerSkipped = True
                self.currPlayerIndex = self.getCounterPlayerIndex(self.currPlayerIndex)
                if len(self.board.getPlaceableCoordinates(self.currPlayerIndex)) == 0:
                    self.isGameFinished = True

    def getCounterPlayerIndex(self, playerIndex):
        return self.board.PLAYER_1 if playerIndex == self.board.PLAYER_0 else self.board.PLAYER_0
    
    def getCurrPlayer(self):
        return self.players[self.currPlayerIndex]

if __name__ == "__main__":
    from train.network.othelloNetWrapper import OthelloNetWrapper
    from train.network.PQNetWrapper import PQNetWrapper
    from train.othelloGameWrapper import OthelloGameWrapper
    from player.randomPlayer import RandomPlayer
    from player.alphaBetaPruningPlayer import AlphaBetaPruningPlayer
    from player.AIPruningPlayer import AIPruningPlayer
    from player.aiPlayer import AIPlayer
    from player.dummyPlayer import DummyPlayer
    import torch
    from train.network.othelloNetWrapper import OthelloNetWrapper
    from trainTasks import tasks

    taskIdx = 4
    ModelType = OthelloNetWrapper
    model = ModelType(OthelloGameWrapper(tasks[taskIdx]))
    model.load_checkpoint(folder= "./model", filename= "(8, 8)_cross_OthelloFCNNet.tar")
    # model.load_checkpoint(folder= "./model", filename= "{'boardSize': (8, 8), 'mode': 'default', 'blockPosType': 'cross'}_PQFCNNet_blip.tar")
    # model.prepareNextTask(3)
    model.setCurrTask(0)
    player0 = AIPlayer(OthelloGameWrapper(tasks[taskIdx]), model)
    player1 = RandomPlayer()

    import random
    pos = [(row,col) for col in range(3) for row in range(3)]
    # pos = [(0,1), (1,2), (2,2)]
    shuffle(pos)
    blocks = []
    for i in range(3):
        blocks.append(pos[i])
        blocks.append((pos[i][0], 7 - pos[i][1]))
        blocks.append((7 - pos[i][0], pos[i][1]))
        blocks.append((7 - pos[i][0], 7 - pos[i][1]))
    board = Board((8,8))
    board.setBlock(blocks)
    board.printBoard()

    wins = [0,0]
    for i in range(100):
        board = Board((8,8), mode="conway")
        board.setBlock(blocks)
        game = Game(board, (player0, player1))
        winner = game.play(printBoard= False)
        if winner != None:
            wins[winner] = wins[winner] + 1
        print(wins)

            



            
    