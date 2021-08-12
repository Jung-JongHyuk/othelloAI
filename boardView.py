from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import *
from boardWidget import BoardWidget

class BoardView(QWidget):
    def __init__(self, boardSize):
        super().__init__()
        self.icons = ["🔵", "🟠", "", "🪨", "⚫️"]
        self.placeableIcons = ["🔹", "🔸"]
        self.boardSize = boardSize
        self.scoreBoardToGridRatio = 10
        self.boardWidget = BoardWidget(self.boardSize, self.getProperGridSize())
        self.playerNameLabels = [QLabel("1", self), QLabel("2", self)]
        self.playerScoreLabels = [QLabel("3", self), QLabel("4", self)]
        with open("./boardView.qss", 'r') as qss:
            self.setStyleSheet(qss.read())
        self.initView()
        self.show()

    def initView(self):
        (rowSize, colSize) = self.boardSize
        #init UI element
        self.setWindowTitle("Othello")

        #set score board layout
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
        
        #set window layout
        mainLayout = QGridLayout()
        mainLayout.addLayout(statusLayout, 0, 0)
        mainLayout.setRowStretch(0, 1)
        mainLayout.addWidget(self.boardWidget, 1, 0)
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
    
    def setGridClickedEventHandler(self, pos, handler):
        (row, col) = pos
        self.boardWidget.grids[row][col].clicked.connect(handler)

    def setGridText(self, pos, text):
        self.boardWidget.setGridText(pos, text)

    def setGridIsUnplaceable(self, pos, isPlaceable):
        self.boardWidget.setGridIsUnplaceable(pos, isPlaceable)

    def setGridIsPlaced(self, pos, isPlaced):
        self.boardWidget.setGridIsPlaced(pos, isPlaced)

    def setGridIsChanged(self, pos, isChanged):
        self.boardWidget.setGridIsChanged(pos, isChanged)
    
    def setGridClickEnabled(self, pos, clickEnabled):
        self.boardWidget.setGridClickEnabled(pos, clickEnabled)
    
    def popAlert(self, message):
        msgBox = QMessageBox()
        msgBox.setText(message)
        msgBox.exec_()

