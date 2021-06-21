import board

class PlayerInterface:
    def __init__(self):
        self.playerIndex = None
    
    def setPlayerIndex(self, playerIndex):
        self.playerIndex = playerIndex

    def decide(self, board):
        pass