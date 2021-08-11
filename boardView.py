from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QLayout, QMessageBox, QVBoxLayout, QWidget, QPushButton, QGridLayout

class BoardView(QWidget):
    def __init__(self, boardSize):
        super().__init__()
        self.icons = ["🔵", "🟠", "", "🪨", "⚫️"]
        self.placeableIcons = ["🔹", "🔸"]
        self.boardSize = boardSize
        self.grids = None
        self.scoreBoardToGridRatio = 10
        self.playerNameLabels = [QLabel("1", self), QLabel("2", self)]
        self.playerScoreLabels = [QLabel("3", self), QLabel("4", self)]
        self.initView()
        self.show()

    def initView(self):
        (rowSize, colSize) = self.boardSize
        #init UI element
        self.setWindowTitle("Othello")
        self.grids = [[QPushButton('', self) for col in range(colSize)] for row in range(rowSize)]
        for row in range(rowSize):
            for col in range(colSize):
                self.grids[row][col].setMaximumHeight(500)

        #set layout
        gridSize = self.getProperGridSize()
        nameLabelSizeRatio = colSize / 2 - 1
        statusLayout = QGridLayout()
        for nameLabels in self.playerNameLabels:
            nameLabels.setFixedWidth(gridSize * nameLabelSizeRatio)
        for scoreLabels in self.playerScoreLabels:
            scoreLabels.setFixedWidth(gridSize)
        statusLayout.addWidget(self.playerNameLabels[0], 0, 0)
        statusLayout.setColumnStretch(0, colSize / 2 - 1)
        statusLayout.addWidget(self.playerScoreLabels[0], 0, 1)
        statusLayout.setColumnStretch(1, 1)
        statusLayout.addWidget(self.playerScoreLabels[1], 0, 2)
        statusLayout.setColumnStretch(2, 1)
        statusLayout.addWidget(self.playerNameLabels[1], 0, 3)
        statusLayout.setColumnStretch(3, colSize / 2 - 1)

        gridLayout = QGridLayout()
        gridLayout.setContentsMargins(1,1,1,1)
        for row in range(rowSize):
            for col in range(colSize):
                self.grids[row][col].setFixedWidth(gridSize)
                self.grids[row][col].setFixedHeight(gridSize)
                gridLayout.addWidget(self.grids[row][col], row + 1, col)
        
        mainLayout = QGridLayout()
        mainLayout.addLayout(statusLayout, 0, 0)
        mainLayout.setRowStretch(0, 1)
        mainLayout.addLayout(gridLayout, 1, 0)
        mainLayout.setRowStretch(1, self.scoreBoardToGridRatio)
        self.setLayout(mainLayout)
    
    def getProperGridSize(self):
        (rowSize, colSize) = self.boardSize
        screenSize = QApplication.primaryScreen().size()
        (screenHeight, screenWidth) = (screenSize.height() * 0.8, screenSize.width() * 0.8)
        if self.scoreBoardToGridRatio * screenHeight / (1 + self.scoreBoardToGridRatio) < screenWidth:
            return self.scoreBoardToGridRatio * screenHeight / ((1 + self.scoreBoardToGridRatio) * colSize)
        else:
            return screenWidth / colSize
    
    def setPlayerName(self, playerIdx, name):
        self.playerNameLabels[playerIdx].setText(str(name))
    
    def setPlayerScore(self, playerIdx, score):
        self.playerScoreLabels[playerIdx].setText(str(score))

    def setGridText(self, pos, text):
        (row, col) = pos
        self.grids[row][col].setText(text)

    def setGridIsUnplaceable(self, pos, isPlaceable):
        (row, col) = pos
        if isPlaceable == True:
            self.grids[row][col].setObjectName("unplaceable")
        self.updateWidgetStyle(self.grids[row][col])

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

