# class Board():

#     # list of all 8 directions on the board, as (x,y) offsets
#     __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

#     def __init__(self, n):
#         "Set up initial board configuration."

#         self.n = n
#         # Create the empty board array.
#         self.pieces = [None]*self.n
#         for i in range(self.n):
#             self.pieces[i] = [0]*self.n

#         # Set up the initial 4 pieces.
#         self.pieces[int(self.n/2)-1][int(self.n/2)] = 1
#         self.pieces[int(self.n/2)][int(self.n/2)-1] = 1
#         self.pieces[int(self.n/2)-1][int(self.n/2)-1] = -1;
#         self.pieces[int(self.n/2)][int(self.n/2)] = -1;

#     # add [][] indexer syntax to the Board
#     def __getitem__(self, index): 
#         return self.pieces[index]

#     def countDiff(self, color):
#         """Counts the # pieces of the given color
#         (1 for white, -1 for black, 0 for empty spaces)"""
#         count = 0
#         for y in range(self.n):
#             for x in range(self.n):
#                 if self[x][y]==color:
#                     count += 1
#                 if self[x][y]==-color:
#                     count -= 1
#         return count

#     def get_legal_moves(self, color):
#         """Returns all the legal moves for the given color.
#         (1 for white, -1 for black
#         """
#         moves = set()  # stores the legal moves.

#         # Get all the squares with pieces of the given color.
#         for y in range(self.n):
#             for x in range(self.n):
#                 if self[x][y]==color:
#                     newmoves = self.get_moves_for_square((x,y))
#                     moves.update(newmoves)
#         return list(moves)

#     def has_legal_moves(self, color):
#         for y in range(self.n):
#             for x in range(self.n):
#                 if self[x][y]==color:
#                     newmoves = self.get_moves_for_square((x,y))
#                     if len(newmoves)>0:
#                         return True
#         return False

#     def get_moves_for_square(self, square):
#         """Returns all the legal moves that use the given square as a base.
#         That is, if the given square is (3,4) and it contains a black piece,
#         and (3,5) and (3,6) contain white pieces, and (3,7) is empty, one
#         of the returned moves is (3,7) because everything from there to (3,4)
#         is flipped.
#         """
#         (x,y) = square

#         # determine the color of the piece.
#         color = self[x][y]

#         # skip empty source squares.
#         if color==0:
#             return None

#         # search all possible directions.
#         moves = []
#         for direction in self.__directions:
#             move = self._discover_move(square, direction)
#             if move:
#                 # print(square,move,direction)
#                 moves.append(move)

#         # return the generated move list
#         return moves

#     def execute_move(self, move, color):
#         """Perform the given move on the board; flips pieces as necessary.
#         color gives the color pf the piece to play (1=white,-1=black)
#         """

#         #Much like move generation, start at the new piece's square and
#         #follow it on all 8 directions to look for a piece allowing flipping.

#         # Add the piece to the empty square.
#         # print(move)
#         flips = [flip for direction in self.__directions
#                       for flip in self._get_flips(move, direction, color)]
#         assert len(list(flips))>0
#         for x, y in flips:
#             #print(self[x][y],color)
#             self[x][y] = color

#     def _discover_move(self, origin, direction):
#         """ Returns the endpoint for a legal move, starting at the given origin,
#         moving by the given increment."""
#         x, y = origin
#         color = self[x][y]
#         flips = []

#         for x, y in Board._increment_move(origin, direction, self.n):
#             if self[x][y] == 0:
#                 if flips:
#                     # print("Found", x,y)
#                     return (x, y)
#                 else:
#                     return None
#             elif self[x][y] == color:
#                 return None
#             elif self[x][y] == -color:
#                 # print("Flip",x,y)
#                 flips.append((x, y))

#     def _get_flips(self, origin, direction, color):
#         """ Gets the list of flips for a vertex and direction to use with the
#         execute_move function """
#         #initialize variables
#         flips = [origin]

#         for x, y in Board._increment_move(origin, direction, self.n):
#             #print(x,y)
#             if self[x][y] == 0:
#                 return []
#             if self[x][y] == -color:
#                 flips.append((x, y))
#             elif self[x][y] == color and len(flips) > 0:
#                 #print(flips)
#                 return flips

#         return []

#     @staticmethod
#     def _increment_move(move, direction, n):
#         # print(move)
#         """ Generator expression for incrementing moves """
#         move = list(map(sum, zip(move, direction)))
#         #move = (move[0]+direction[0], move[1]+direction[1])
#         while all(map(lambda x: 0 <= x < n, move)): 
#         #while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
#             yield move
#             move=list(map(sum,zip(move,direction)))
#             #move = (move[0]+direction[0],move[1]+direction[1])

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

    def printBoard(self):
        printLiterals = ["O", "#", " ", "*"]
        print("  ", end = "")
        for i in range(self.colSize):
            print(i, end = " ")
        print("")
        for rowIndex in range(self.rowSize):
            for colIndex in range(self.colSize):
                if colIndex == 0:
                    print(rowIndex, end = " ")
                print(printLiterals[self.board[rowIndex][colIndex]], end = " ")
            print("")


