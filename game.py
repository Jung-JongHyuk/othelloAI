from board import Board
from player.playerInterface import PlayerInterface

class Game:
    def __init__(self, player0, player1, board):
        player0.setPlayerIndex(0)
        player1.setPlayerIndex(1)
        self.players = [player0, player1]
        self.board = board
        self.currPlayer = self.board.PLAYER_0
    
    # simulate entire game and return winner. If draw, return None.
    def play(self):
        boardStatus = self.board.getBoardStatus()
        isPrevPlayerSkipped = False
        while boardStatus[-1] > 0:
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

        if boardStatus[self.board.PLAYER_0] > boardStatus[self.board.PLAYER_1]:
            return self.board.PLAYER_0
        elif boardStatus[self.board.PLAYER_0] < boardStatus[self.board.PLAYER_1]:
            return self.board.PLAYER_1
        else:
            return None


if __name__ == "__main__":
    from player.randomPlayer import RandomPlayer

    player0 = RandomPlayer()
    player1 = RandomPlayer()
    wins = [0,0]
    for i in range(100):
        game = Game(player0, player1, Board((8,8), 5))
        winner = game.play()
        if winner != None:
            wins[winner] = wins[winner] + 1
    print(wins) 
            



            
    