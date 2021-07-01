from board import Board
from game import Game

class BoardUIController:
    def __init__(self, board, ui):
        self.board = board
        self.ui = ui

    def placePieceOnBoard(self, pos):
        self.board.placePiece(pos, self.board.currentTurnPlayer)
        self.ui.updateUI()