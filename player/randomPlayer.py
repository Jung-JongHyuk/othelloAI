import random
from player.playerInterface import PlayerInterface

class RandomPlayer(PlayerInterface):
    def __init__(self):
        super().__init__()

    def decide(self, board):
        placeablePos = board.getPlaceableCoordinates(self.playerIndex)
        if len(placeablePos) == 0:
            return None
        else:
            return random.choice(placeablePos)