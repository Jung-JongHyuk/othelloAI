from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout

class BoardView(QWidget):
    def __init__(self, boardSize):
        super().__init__()
        self.icons = ["🔵", "🟠", "", "❌"]
        self.placeableIcons = ["🔹", "🔸"]
        self.boardSize = boardSize
        self.grids = []
        self.initView()

    def initView(self):
        (rowSize, colSize) = self.boardSize
        #init UI element
        self.setWindowTitle("Othello")
        self.grids = [[QPushButton('', self) for col in range(colSize)] for row in range(rowSize)]
        for row in range(rowSize):
            for col in range(colSize):
                self.grids[row][col].setStyleSheet("QPushButton {background-color: rgba(0,100,0,0.1); font-size: 50px; width: 50px; height: 50px}")
                self.grids[row][col].setMaximumHeight(500)
                # self.grids[row][col].setMaximumWidth(150)

        #set layout
        layout = QGridLayout()
        for row in range(rowSize):
            for col in range(colSize):
                layout.addWidget(self.grids[row][col], row, col)
        self.setLayout(layout)
        self.setGeometry(0, 0, 1000, 1000)
        self.show()

    def setGridText(self, pos, text):
        (row, col) = pos
        self.grids[row][col].setText(text)

    def setGridIsPlaced(self, pos, isPlaced):
        (row, col) = pos
        if isPlaced == True:
            self.grids[row][col].setStyleSheet("QPushButton {background-color: rgba(0,100,0,0.5); font-size: 50px; width: 50px; height: 50px}")
        else:
            self.grids[row][col].setStyleSheet("QPushButton {background-color: rgba(0,100,0,0.1); font-size: 50px; width: 50px; height: 50px}")

    def setGridIsChanged(self, pos, isChanged):
        (row, col) = pos
        if isChanged == True:
            self.grids[row][col].setStyleSheet("QPushButton {background-color: rgba(0,100,0,0.3); font-size: 50px; width: 50px; height: 50px}")
        else:
            self.grids[row][col].setStyleSheet("QPushButton {background-color: rgba(0,100,0,0.1); font-size: 50px; width: 50px; height: 50px}")
    
    def setGridClickEnabled(self, pos, clickEnabled):
        (row, col) = pos
        self.grids[row][col].setEnabled(clickEnabled)
