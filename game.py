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
                # print(self.currPlayerIndex, "skipped")
                self.isPrevPlayerSkipped = True
                self.currPlayerIndex = self.getCounterPlayerIndex(self.currPlayerIndex)
                if len(self.board.getPlaceableCoordinates(self.currPlayerIndex)) == 0:
                    self.isGameFinished = True

    def getCounterPlayerIndex(self, playerIndex):
        return self.board.PLAYER_1 if playerIndex == self.board.PLAYER_0 else self.board.PLAYER_0
    
    def getCurrPlayer(self):
        return self.players[self.currPlayerIndex]

if __name__ == "__main__":
    from player.randomPlayer import RandomPlayer
    from player.alphaBetaPruningPlayer import AlphaBetaPruningPlayer
    from player.AIPruningPlayer import AIPruningPlayer
    from player.aiPlayer import AIPlayer
    from player.dummyPlayer import DummyPlayer
    import torch
    from train.network.othelloNetWrapper import OthelloNetWrapper

    GPU_NUM = 1
    device = torch.device(f"cuda:{GPU_NUM}") if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)
    boardSize = (8,8)
    ModelType = OthelloNetWrapper
    model = ModelType(OthelloGameWrapper(boardSize, "none"))
    model.setCurrTask(0)
    model.load_checkpoint(folder= "./model", filename= "(6, 6)_none_OthelloFCNNet.tar")
    player0 = AIPlayer(boardSize, model)
    # player1 = AIPruningPlayer((6,6), modelName='best.pth.tar', seachDepth=3)
    # player1 = AIPlayer(boardSize, folderName= './model', modelName='QNetWrapper_(6, 6)_none_checkpoint_15.pth.tar')
    player1 = RandomPlayer()

    wins = [0,0]
    for i in range(100):
        board = Board(boardSize, mode= "conway")
        game = Game(board, (player0, player1))
        winner = game.play(printBoard= False)
        if winner != None:
            wins[winner] = wins[winner] + 1
        print(wins)

            



            
    