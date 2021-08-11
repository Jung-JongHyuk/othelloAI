from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import *

class MainView(QWidget):
    def __init__(self):
        super().__init__()
        # game setting inputs
        self.rowSizeInput = QLineEdit(self)
        self.colSizeInput = QLineEdit(self)
        self.blockPosTypeComboBox = QComboBox(self)
        self.blockPosTypeComboBox.addItems(["none", "x-cross", "cross", "random", "custom"])
        self.gameModeComboBox = QComboBox(self)
        self.gameModeComboBox.addItems(["default", "conway", "desdemona"])
        self.player0ComboBox = QComboBox(self)
        self.player0ComboBox.addItems(["Human", "Computer", "AI"])
        self.player1ComboBox = QComboBox(self)
        self.player1ComboBox.addItems(["Human", "Computer", "AI"])
        self.gameStartButton = QPushButton(self)
        self.gameStartButton.setText("Game Start")
        self.initView()
        self.show()

    def initView(self):
        self.setWindowTitle("Othello")
        mainLayout = QGridLayout()
        image = QPixmap()
        image.load("./othelloImage.jpg")
        imageLabel = QLabel()
        imageLabel.setPixmap(image)
        mainLayout.addWidget(imageLabel, 0, 0)

        settingBox = QGroupBox("Game Setting")

        settingLayout = QGridLayout()
        # board size input
        currRow = 0
        settingLayout.addWidget(QLabel("Board Size"), currRow, 0)
        settingLayout.addWidget(QLabel(":"), currRow, 1)
        boardSizeInputLayout = QGridLayout()
        boardSizeInputLayout.addWidget(self.rowSizeInput, currRow, 0)
        boardSizeInputLayout.addWidget(QLabel("x"), currRow, 1)
        boardSizeInputLayout.addWidget(self.colSizeInput, currRow, 2)
        settingLayout.addLayout(boardSizeInputLayout, currRow, 2)
        settingLayout.setRowStretch(currRow, 2)

        # block pos type input
        currRow = 1
        settingLayout.addWidget(QLabel("Block Type"), currRow, 0)
        settingLayout.addWidget(QLabel(":"), currRow, 1)
        settingLayout.addWidget(self.blockPosTypeComboBox, currRow, 2)
        settingLayout.setRowStretch(currRow, 2)

        # game mode input
        currRow = 2
        settingLayout.addWidget(QLabel("Game mode"), currRow, 0)
        settingLayout.addWidget(QLabel(":"), currRow, 1)
        settingLayout.addWidget(self.gameModeComboBox, currRow, 2)
        settingLayout.setRowStretch(currRow, 2)

        # player input
        currRow = 3
        settingLayout.addWidget(QLabel("Player 0"), currRow, 0)
        settingLayout.addWidget(QLabel(":"), currRow, 1)
        settingLayout.addWidget(self.player0ComboBox, currRow, 2)
        settingLayout.setRowStretch(currRow, 1)
        currRow = 4
        settingLayout.addWidget(QLabel("Player 1"), currRow, 0)
        settingLayout.addWidget(QLabel(":"), currRow, 1)
        settingLayout.addWidget(self.player1ComboBox, currRow, 2)
        settingLayout.setRowStretch(currRow, 1)

        currRow = 5
        settingLayout.addWidget(self.gameStartButton, currRow, 0, 1, 3)
        settingLayout.setRowStretch(currRow, 1)
        settingBox.setLayout(settingLayout)
        mainLayout.addWidget(settingBox, 0, 1)
        self.setLayout(mainLayout)
        self.setMaximumWidth(0) # set window width to minimum
    
    def getInput(self):
        kwargs = {}
        kwargs["player0"] = self.player0ComboBox.currentText()
        kwargs["player1"] = self.player1ComboBox.currentText()
        kwargs["boardSetting"] = {}
        kwargs["boardSetting"]["boardSize"] = (int(self.rowSizeInput.text()), int(self.colSizeInput.text()))
        kwargs["boardSetting"]["blockPosType"] = self.blockPosTypeComboBox.currentText()
        kwargs["boardSetting"]["mode"] = self.gameModeComboBox.currentText()

        return kwargs

if __name__ == "__main__":
    app = QApplication([])
    mainView = MainView()
    mainView.show()
    app.exec_()