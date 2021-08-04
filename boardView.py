from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMessageBox, QVBoxLayout, QWidget, QPushButton, QGridLayout

class BoardView(QWidget):
    def __init__(self, boardSize):
        super().__init__()
        self.icons = ["🔵", "🟠", "", "❌", "⚫️"]
        self.placeableIcons = ["🔹", "🔸"]
        self.boardSize = boardSize
        self.grids = None
        self.playerNameLabels = [QLabel("1", self), QLabel("2", self)]
        self.playerScoreLabels = [QLabel("3", self), QLabel("4", self)]
        self.initView()

    def initView(self):
        (rowSize, colSize) = self.boardSize
        #init UI element
        self.setWindowTitle("Othello")
        self.grids = [[QPushButton('', self) for col in range(colSize)] for row in range(rowSize)]
        for row in range(rowSize):
            for col in range(colSize):
                self.grids[row][col].setMaximumHeight(500)

        #set layout
        statusLayout = QGridLayout()
        statusLayout.addWidget(self.playerNameLabels[0], 0, 0)
        statusLayout.setColumnStretch(0, colSize / 2 - 1)
        statusLayout.addWidget(self.playerScoreLabels[0], 0, 1)
        statusLayout.setColumnStretch(1, 1)
        statusLayout.addWidget(self.playerScoreLabels[1], 0, 2)
        statusLayout.setColumnStretch(2, 1)
        statusLayout.addWidget(self.playerNameLabels[1], 0, 3)
        statusLayout.setColumnStretch(3, colSize / 2 - 1)

        gridLayout = QGridLayout()
        for row in range(rowSize):
            for col in range(colSize):
                gridLayout.addWidget(self.grids[row][col], row + 1, col)
        
        mainLayout = QGridLayout()
        mainLayout.addLayout(statusLayout, 0, 0)
        mainLayout.setRowStretch(0, 1)
        mainLayout.addLayout(gridLayout, 1, 0)
        mainLayout.setRowStretch(1, 10)
        self.setLayout(mainLayout)
        self.setGeometry(0, 0, 1000, 1000)
        self.show()
    
    def setPlayerName(self, playerIdx, name):
        self.playerNameLabels[playerIdx].setText(str(name))
    
    def setPlayerScore(self, playerIdx, score):
        self.playerScoreLabels[playerIdx].setText(str(score))

    def setGridText(self, pos, text):
        (row, col) = pos
        self.grids[row][col].setText(text)

    def setGridIsPlaced(self, pos, isPlaced):
        (row, col) = pos
        if isPlaced == True:
            self.grids[row][col].setObjectName("placed")
        else:
            self.grids[row][col].setObjectName("")
        self.updateWidgetStyle(self.grids[row][col])

    def setGridIsChanged(self, pos, isChanged):
        (row, col) = pos
        if isChanged == True:
            self.grids[row][col].setObjectName("changed")
        else:
            self.grids[row][col].setObjectName("")
        self.updateWidgetStyle(self.grids[row][col])
    
    def setGridClickEnabled(self, pos, clickEnabled):
        (row, col) = pos
        self.grids[row][col].setEnabled(clickEnabled)
    
    def updateWidgetStyle(self, widget):
        widget.style().polish(widget)
        widget.style().unpolish(widget)
        widget.update()
    
    def popAlert(self, message):
        msgBox = QMessageBox()
        msgBox.setText(message)
        msgBox.exec_()

