import random
import sys

class Board:
    def __init__(self, boardSize, mode= "default", blockPosType= "none", numOfBlock= 0, customBlockPos= []):
        (self.rowSize, self.colSize, self.blockPosType, self.numOfBlock, self.mode) = (boardSize[0], boardSize[1], blockPosType, numOfBlock, mode)
        (self.PLAYER_0, self.PLAYER_1, self.VOID, self.BLOCK, self.NEUTRAL) = (0, 1, 2, 3, 4)
        self.board = [[self.VOID for col in range(self.colSize)] for row in range(self.rowSize)]
        self.placeInitialPiece()
        if blockPosType == "random":
            self.makeRandomBlock(numOfBlock)
        else:
            blockPos = []
            if blockPosType == "x-cross":
                blockPos = self.makeXCrossBlockPos()
            elif blockPosType == "cross":
                blockPos = self.makeCrossBlockPos()
            elif blockPosType == "octagon":
                blockPos = self.makeOctagonBlockPos()
            elif blockPosType == "custom":
                blockPos = self.makeSymmetricBlockPos(customBlockPos)

            elif blockPosType != "none":
                print("invalid blockPosType")
                exit()
            self.setBlock(blockPos)
        if self.mode not in ["default", "conway", "desdemona"]:
            print("invalid mode")
            exit()

    def makeCrossBlockPos(self):
        blockPos = []
        crossLength = int((min(self.rowSize, self.colSize) - 4) / 2)
        (midRowIdx, midColIdx) = (int(self.rowSize / 2), int(self.colSize / 2))
        for i in range(crossLength):
            blockPos.append((i, midColIdx - 1))
            blockPos.append((i, midColIdx))
            blockPos.append((midRowIdx - 1, i))
            blockPos.append((midRowIdx, i))
            blockPos.append((self.rowSize - i - 1, midColIdx - 1))
            blockPos.append((self.rowSize - i - 1, midColIdx))
            blockPos.append((midRowIdx - 1, self.colSize - i - 1))
            blockPos.append((midRowIdx, self.colSize - i - 1))
        return blockPos

    def makeXCrossBlockPos(self):
        blockPos = []
        crossLength = int((min(self.rowSize, self.colSize) - 2) / 2)
        for i in range(crossLength):
            blockPos.append((i, i))
            blockPos.append((i, self.colSize - i - 1))
            blockPos.append((self.rowSize - i - 1, i))
            blockPos.append((self.rowSize - i - 1, self.colSize - i - 1))
        return blockPos
    
    def makeOctagonBlockPos(self):
        blockPos = []
        crossLength = int(min(self.rowSize, self.colSize) / 2)
        for i in range(crossLength):
            blockPos.append((int(self.rowSize / 2) - i - 1, i))
            blockPos.append((int(self.rowSize / 2) + i, i))
            blockPos.append((self.rowSize - i - 1, int(self.colSize / 2 + i)))
            blockPos.append((int(self.rowSize / 2) - i - 1, self.colSize - i - 1))
        return blockPos
    
    def makeSymmetricBlockPos(self, customBlockPos):
        blockPos = []
        for pos in customBlockPos:
            blockPos.append(pos)
            blockPos.append((pos[0], self.colSize - pos[1] - 1))
            blockPos.append((self.rowSize - pos[0] - 1, pos[1]))
            blockPos.append((self.rowSize - pos[0] - 1, self.colSize - pos[1] - 1))
        return blockPos

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

    def makeRandomBlock(self):
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
            elif self.mode == "desdemona" and self.board[currentRowIndex][currentColIndex] == self.NEUTRAL:
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
                    if self.mode == "desdemona":
                        self.board[currentRowIndex][currentColIndex] = playerIndex if self.board[currentRowIndex][currentColIndex] == self.NEUTRAL else self.NEUTRAL
                    else:
                        self.board[currentRowIndex][currentColIndex] = playerIndex
                    (currentRowIndex, currentColIndex) = (currentRowIndex + rowMoveDirections[i], currentColIndex + colMoveDirections[i])
        self.board[rowIndex][colIndex] = playerIndex

        if self.mode == "conway":
            nextBoard = self.board
            for row in range(self.rowSize):
                for col in range(self.colSize):
                    pieceCount = self.countAdjPiece((row,col))
                    if self.board[row][col] == self.PLAYER_0 and pieceCount[self.PLAYER_1] > sum(pieceCount[:-1]) / 2:
                        nextBoard[row][col] = self.PLAYER_1
                    elif self.board[row][col] == self.PLAYER_1 and pieceCount[self.PLAYER_0] > sum(pieceCount[:-1]) / 2:
                        nextBoard[row][col] = self.PLAYER_0
            self.board = nextBoard

    def countAdjPiece(self, pos):
        rowMoveDirections = [-1, -1, 0, 1, 1, 1, 0, -1]
        colMoveDirections = [0, 1, 1, 1, 0, -1, -1, -1]
        pieceCount = [0,0,0,0]
        for (drow, dcol) in zip(rowMoveDirections, colMoveDirections):
            adjPos = (pos[0] + drow, pos[1] + dcol)
            if not self.isIndexOutOfRange(adjPos[0], adjPos[1]):
                pieceCount[self.board[adjPos[0]][adjPos[1]]] = pieceCount[self.board[adjPos[0]][adjPos[1]]] + 1
        return pieceCount
    
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
    board = Board((6,6), 'octagon')
    board.printBoard()


