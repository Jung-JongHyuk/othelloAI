import random
import sys

class Board:
    def __init__(self, boardSize, numOfBlock):
        (self.rowSize, self.colSize, self.numOfBlock) = (boardSize[0], boardSize[1], numOfBlock)
        (self.PLAYER_0, self.PLAYER_1, self.VOID, self.BLOCK) = (0, 1, 2, 3)
        self.board = [[self.VOID for col in range(self.colSize)] for row in range(self.rowSize)]
        # self.currentTurnPlayer = self.PLAYER_0
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

    #사용자 지정 위치에 block
    def setBlock(self, blockCoordinates):
        self.eraseBlock()
        for coordinate in blockCoordinates:
            self.board[coordinate[0]][coordinate[1]] = self.BLOCK
    
    def eraseBlock(self):
        for rowIndex in range(self.rowSize):
            for colIndex in range(self.colSize):
                if self.board[rowIndex][colIndex] == self.BLOCK:
                    self.board[rowIndex][colIndex] = self.VOID

    def counterPlayerIndex(self, playerIndex):
        return self.PLAYER_1 if playerIndex == self.PLAYER_0 else self.PLAYER_0
        
    def getPlaceableCoordinates(self, playerIndex):
        rowMoveDirections = [-1, -1, 0, 1, 1, 1, 0, -1]
        colMoveDirections = [0, 1, 1, 1, 0, -1, -1, -1]
        possibleCoordinates = []
        for rowIndex in range(self.rowSize):
            for colIndex in range(self.colSize):
                if self.board[rowIndex][colIndex] == self.VOID:
                    for i in range(8):
                        if self.findWallCoordinate(rowIndex, colIndex, rowMoveDirections[i], colMoveDirections[i], playerIndex) != None:
                            possibleCoordinates.append((rowIndex, colIndex))
                            break
        return possibleCoordinates

    def findWallCoordinate(self, rowIndex, colIndex, rowMoveDirection, colMoveDirection, playerIndex):
        (currentRowIndex, currentColIndex) = (rowIndex + rowMoveDirection, colIndex + colMoveDirection)
        numOfCounterPlayerPieceBetweenWall = 0
        while not self.isIndexOutOfRange(currentRowIndex, currentColIndex):
            if self.board[currentRowIndex][currentColIndex] == self.counterPlayerIndex(playerIndex):
                numOfCounterPlayerPieceBetweenWall = numOfCounterPlayerPieceBetweenWall + 1
            elif self.board[currentRowIndex][currentColIndex] == playerIndex:
                if numOfCounterPlayerPieceBetweenWall > 0:
                    return (currentRowIndex, currentColIndex)
                else:
                    return None
            else:
                return None
            (currentRowIndex, currentColIndex) = (currentRowIndex + rowMoveDirection, currentColIndex + colMoveDirection)
        
    def isIndexOutOfRange(self, rowIndex, colIndex):
        return rowIndex < 0 or rowIndex >= self.rowSize or colIndex < 0 or colIndex >= self.colSize

    def placePiece(self, posToPlace, playerIndex):
        rowMoveDirections = [-1, -1, 0, 1, 1, 1, 0, -1]
        colMoveDirections = [0, 1, 1, 1, 0, -1, -1, -1]
        wallCoordinates = []
        (rowIndex, colIndex) = posToPlace
        for i in range(8):
            wallCoordinates.append(self.findWallCoordinate(rowIndex, colIndex, rowMoveDirections[i], colMoveDirections[i], playerIndex))
        for i in range(8):
            if wallCoordinates[i] != None:
                (currentRowIndex, currentColIndex) = (rowIndex + rowMoveDirections[i], colIndex + colMoveDirections[i])
                while (currentRowIndex, currentColIndex) != wallCoordinates[i]:
                    self.board[currentRowIndex][currentColIndex] = playerIndex
                    (currentRowIndex, currentColIndex) = (currentRowIndex + rowMoveDirections[i], currentColIndex + colMoveDirections[i])
        self.board[rowIndex][colIndex] = playerIndex
        # self.currentTurnPlayer = self.counterPlayerIndex(playerIndex)
    
    # return board status [player0 piece size, player1 piece size, ... void size]
    def getBoardStatus(self):
        status = [0, 0, 0]
        for row in range(self.rowSize):
            for col in range(self.colSize):
                if self.board[row][col] == self.PLAYER_0 or self.board[row][col] == self.PLAYER_1:
                    status[self.board[row][col]] = status[self.board[row][col]] + 1
                elif self.board[row][col] == self.VOID:
                    status[-1] = status[-1] + 1
        return status

    def printBoard(self, printLabel= False):
        printLiterals = ["O", "#", " ", "*"]
        for i in range(self.colSize * 2):
            print("-", end= "")
        print("")

        if printLabel == True:
            print("  ", end = "")
            for i in range(self.colSize):
                print(i, end = " ")
            print("")
        for rowIndex in range(self.rowSize):
            for colIndex in range(self.colSize):
                if colIndex == 0 and printLabel == True:
                    print(rowIndex, end = " ")
                print(printLiterals[self.board[rowIndex][colIndex]], end = " ")
            print("")
        
        for i in range(self.colSize * 2):
            print("-", end= "")
        print("")

if __name__ == "__main__":
    board = Board((8,8),5)
    board.printBoard()

    while True:
        print("currentPlayer: ", board.currentTurnPlayer)
        board.printBoard()
        print(board.getPlaceableCoordinates(board.currentTurnPlayer))
        coordinateX = int(input())
        coordinateY = int(input())
        board.placePiece(coordinateX, coordinateY, board.currentTurnPlayer)


