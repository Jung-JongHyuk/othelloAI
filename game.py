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
    def play(self):
        boardStatus = self.board.getBoardStatus()
        self.isPrevPlayerSkipped = False
        self.isGameFinished = False
        while not self.isGameFinished:
            posToPlace = self.getCurrPlayer().decide(self.board)
            self.adoptDecision(posToPlace)
            boardStatus = self.board.getBoardStatus()
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
            self.currPlayerIndex = self.getCounterPlayerIndex()
            if len(self.board.getPlaceableCoordinates(self.currPlayerIndex)) == 0:
                print(self.currPlayerIndex, "skipped")
                self.isPrevPlayerSkipped = True
                self.currPlayerIndex = self.getCounterPlayerIndex()
                if len(self.board.getPlaceableCoordinates(self.currPlayerIndex)) == 0:
                    self.isGameFinished = True

    def getCounterPlayerIndex(self):
        return self.board.PLAYER_1 if self.currPlayerIndex == self.board.PLAYER_0 else self.board.PLAYER_0
    
    def getCurrPlayer(self):
        return self.players[self.currPlayerIndex]

if __name__ == "__main__":
    from player.randomPlayer import RandomPlayer

    player0 = RandomPlayer()
    player1 = RandomPlayer()
    wins = [0,0]
    for i in range(1):
        game = Game(Board((8,8), 5), (player0, player1))
        winner = game.play()
        if winner != None:
            wins[winner] = wins[winner] + 1
    print(wins) 
            



            
    