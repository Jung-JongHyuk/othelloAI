from board import Board
from player.playerInterface import PlayerInterface

class Game:
    def __init__(self, player0, player1, boardSize, numOfBlock):
        player0.setPlayerIndex(0)
        player1.setPlayerIndex(1)
        self.players = [player0, player1]
        self.board = Board(boardSize, numOfBlock)
        self.currPlayer = self.board.PLAYER_0
    
    def start(self):
        boardStatus = self.board.getBoardStatus()
        isPrevPlayerSkipped = False
        while boardStatus["void"] > 0:
            posToPlace = self.players[self.currPlayer].decide(self.board)
            if posToPlace != None:
                self.board.placePiece(posToPlace, self.currPlayer)
                isPrevPlayerSkipped = False
            elif isPrevPlayerSkipped:
                break
            else:
                isPrevPlayerSkipped = True
            boardStatus = self.board.getBoardStatus()
            self.currPlayer = self.board.counterPlayerIndex(self.currPlayer)
            self.board.printBoard()
            print(boardStatus)


if __name__ == "__main__":
    from player.randomPlayer import RandomPlayer

    player0 = RandomPlayer()
    player1 = RandomPlayer()
    game = Game(player0, player1, (8,8), 5)
    game.start()
            



            
    