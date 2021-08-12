from PyQt5.QtWidgets import *

class BoardWidget(QWidget):
    def __init__(self, boardSize, gridSize):
        super().__init__()
        self.icons = ["🔵", "🟠", "", "🪨", "⚫️"]
        self.placeableIcons = ["🔹", "🔸"]
        self.boardSize = boardSize
        self.gridSize = gridSize
        self.grids = None
        with open("./boardWidget.qss", 'r') as qss:
            self.setStyleSheet(qss.read())
        self.initView()
        self.show()
    
    def initView(self):
        (rowSize, colSize) = self.boardSize
        self.grids = [[QPushButton('', self) for col in range(colSize)] for row in range(rowSize)]
        gridLayout = QGridLayout()
        for row in range(rowSize):
            for col in range(colSize):
                self.grids[row][col].setFixedWidth(self.gridSize)
                self.grids[row][col].setFixedHeight(self.gridSize)
                gridLayout.addWidget(self.grids[row][col], row, col)
        self.setLayout(gridLayout)
    
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

