import logging
import math
from pickle import Pickler, Unpickler
import gc

from train.Coach import Coach, logFilePostfix
from train.othelloGameWrapper import OthelloGameWrapper
from train.network.othelloNetWrapper import OthelloNetWrapper
from train.network.QNetWrapper import QNetWrapper
from train.network.PQNetWrapper import PQNetWrapper
from train.utils import *
from train.blip_utils import *
from trainTasks import tasks
from game import Game
from board import Board
from player.randomPlayer import RandomPlayer
from player.alphaBetaPruningPlayer import AlphaBetaPruningPlayer
from player.AIPruningPlayer import AIPruningPlayer
from player.aiPlayer import AIPlayer

def getNumOfParam(model):
    return sum(p.numel() for p in model.nnet.parameters())

GPU_NUM = 2
ModelType = PQNetWrapper
device = torch.device(f"cuda:{GPU_NUM}") if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)
game = OthelloGameWrapper(tasks[0])
model = ModelType(game)
for (task, param) in enumerate(tasks):
    if isinstance(model, OthelloNetWrapper):
        fileName = f"{tasks[task]}_{type(model.nnet).__name__}.tar"
    else:
        fileName = f"{tasks[task]}_{type(model.nnet).__name__}_blip.tar"
    model.load_checkpoint(folder= './model_256', filename= fileName)
    print(getNumOfParam(model))

print(sum(p.numel() for p in model.nnet.conv3.layers[0].parameters()))