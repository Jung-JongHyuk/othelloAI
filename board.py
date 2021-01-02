import random
import sys

class Board:
    def __init__(self, rowSize, colSize, numOfBlock):
        (self.rowSize, self.colSize, self.numOfBlock) = (rowSize, colSize, numOfBlock)
        (self.PLAYER_0, self.PLAYER_1, self.VOID, self.BLOCK) = (0, 1, 2, 3)
        self.board = [[self.VOID for col in range(colSize)] for row in range(rowSize)]
        self.placeInitialPiece()
        self.makeBlock()

    def placeInitialPiece(self):
        if self.rowSize % 2 != 0 or self.colSize % 2 != 0:
            print("board의 가로 세로 길이는 항상 짝수여야 합니다.")
            sys.exit()
        else:
            self.board[int(self.rowSize / 2) - 1][int(self.colSize / 2) - 1] = self.PLAYER_1
            self.board[int(self.rowSize / 2)][int(self.colSize / 2)] = self.PLAYER_1
            self.board[int(self.rowSize / 2) - 1][int(self.colSize / 2)] = self.PLAYER_0
            self.board[int(self.rowSize / 2)][int(self.colSize / 2) - 1] = self.PLAYER_0

    def isInitialPiecePos(self, rowIndex, colIndex):
        return (rowIndex >= self.rowSize / 2 - 1 and rowIndex <= self.rowSize / 2) and (colIndex >= self.colSize / 2 - 1 and colIndex <= self.colSize / 2)

    def makeBlock(self):
        voidCoordinate = []
        for rowIndex in range(self.rowSize):
            for colIndex in range(self.colSize):
                if self.isInitialPiecePos(rowIndex, colIndex) == False:
                    voidCoordinate.append((rowIndex, colIndex))
        random.shuffle(voidCoordinate)
        for i in range(self.numOfBlock):
            coordinate = voidCoordinate[i]
            self.board[coordinate[0]][coordinate[1]] = self.BLOCK

    def printBoard(self):
        printLiterals = ["O", "#", ".", "*"]
        for rowIndex in range(self.rowSize):
            for colIndex in range(self.colSize):
                print(printLiterals[self.board[rowIndex][colIndex]], end = "")
            print("")

board = Board(8,8,5)
board.printBoard()

