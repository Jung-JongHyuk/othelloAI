from board import Board
from boardViewController import BoardUIController
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout

class BoardView(QWidget):
    def __init__(self):
        super().__init__()
        self.board = Board((8,8),5)
        self.uic = BoardUIController(self.board, self)

        #init UI element
        self.setWindowTitle("Othello")
        self.icons = ["🔵", "🟠", "", "❌"]
        self.placeableIcons = ["🔹", "🔸"]
        self.buttons = [[QPushButton('', self) for col in range(self.board.colSize)] for row in range(self.board.rowSize)]
        for row in range(self.board.rowSize):
            for col in range(self.board.colSize):
                self.buttons[row][col].setStyleSheet("QPushButton {background-color: rgb(255,255,255); font-size: 50px}")
                self.buttons[row][col].setMaximumHeight(500)
                self.buttons[row][col].setMaximumWidth(500)
                self.buttons[row][col].clicked.connect(lambda state, pos=(row,col): self.uic.placePieceOnBoard(pos))

        #set layout
        layout = QGridLayout()
        for row in range(self.board.rowSize):
            for col in range(self.board.colSize):
                layout.addWidget(self.buttons[row][col], row, col)
        self.setLayout(layout)
        self.setGeometry(300, 300, 300, 300)
        self.updateUI()
        self.show()

    def updateUI(self):
        for row in range(self.board.rowSize):
            for col in range(self.board.colSize):
                self.buttons[row][col].setText(self.icons[self.board.board[row][col]])
                self.buttons[row][col].setEnabled(False)

        placeableCoordinates = self.board.getPlaceableCoordinates(self.board.currentTurnPlayer)
        for coordinate in placeableCoordinates:
            self.buttons[coordinate[0]][coordinate[1]].setText(self.placeableIcons[self.board.currentTurnPlayer])
            self.buttons[coordinate[0]][coordinate[1]].setEnabled(True)

if __name__ == '__main__':
    app = QApplication([])
    ex = BoardView()
    app.exec_()