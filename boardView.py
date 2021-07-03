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
                self.grids[row][col].setStyleSheet("QPushButton {background-color: rgb(255,255,255); font-size: 50px}")
                self.grids[row][col].setMaximumHeight(150)
                self.grids[row][col].setMaximumWidth(150)

        #set layout
        layout = QGridLayout()
        for row in range(rowSize):
            for col in range(colSize):
                layout.addWidget(self.grids[row][col], row, col)
        self.setLayout(layout)
        self.setGeometry(0, 0, 500, 500)
        self.show()

    def setGridText(self, pos, text):
        (row, col) = pos
        self.grids[row][col].setText(text)
    
    def setGridClickEnabled(self, pos, clickEnabled):
        (row, col) = pos
        self.grids[row][col].setEnabled(clickEnabled)
