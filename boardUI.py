from board import Board
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.board = Board(8,8,5)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('QPushButton')
        self.buttons = [[QPushButton('', self) for col in range(self.board.colSize)] for row in range(self.board.rowSize)]
        for row in range(self.board.rowSize):
            for col in range(self.board.colSize):
                self.buttons[row][col].setStyleSheet('QPushButton {background-color: rgb(255,255,255); font-size: 50px}')
                self.buttons[row][col].setMaximumHeight(500)
                self.buttons[row][col].setMaximumWidth(500)
                self.buttons[row][col].clicked.connect(self.makePutPiece(row, col))

        layout = QGridLayout()
        for row in range(self.board.rowSize):
            for col in range(self.board.colSize):
                layout.addWidget(self.buttons[row][col], row, col)
        self.setLayout(layout)
        self.setGeometry(300, 300, 300, 200)
        self.updateUI()
        self.show()

    def updateUI(self):
        icons = ["🔵", "🟠", "", "❌"]
        for row in range(self.board.rowSize):
            for col in range(self.board.colSize):
                self.buttons[row][col].setText(icons[self.board.board[row][col]])
                self.buttons[row][col].setEnabled(False)

        placeableIcons = ["🔹", "🔸"]
        placeableCoordinates = self.board.getPlaceableCoordinates(self.board.currentTurnPlayer)
        for coordinate in placeableCoordinates:
            self.buttons[coordinate[0]][coordinate[1]].setText(placeableIcons[self.board.currentTurnPlayer])
            self.buttons[coordinate[0]][coordinate[1]].setEnabled(True)
    
    def makePutPiece(self, rowIndex, colIndex):
        def putPiece():
            self.board.placePiece(rowIndex, colIndex, self.board.currentTurnPlayer)
            self.updateUI()
        return putPiece

if __name__ == '__main__':
    app = QApplication([])
    ex = MyApp()
    app.exec_()