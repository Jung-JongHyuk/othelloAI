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

gc.collect()
torch.cuda.empty_cache()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
while log.hasHandlers():
    log.removeHandler(log.handlers[0])
streamHandler = logging.StreamHandler()
fileHandler = logging.FileHandler(f'fisherLog_{logFilePostfix}.log', 'w')
log.propagate = False
log.addHandler(streamHandler)
log.addHandler(fileHandler)

args = dotdict({
    'numIters': 30,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/','best_PQFCNNet.pth.tar'),
    'resume_iter': 1,
    'numItersForTrainExamplesHistory': 20
})

def main():
    GPU_NUM = 2
    trainBeginTask = 2
    ModelType = PQNetWrapper
    expandThreshold = [1,1,0.8,0.73]

    device = torch.device(f"cuda:{GPU_NUM}") if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)
    game = OthelloGameWrapper(tasks[trainBeginTask])
    model = ModelType(game)
    if trainBeginTask != 0:
        if isinstance(model, OthelloNetWrapper):
            fileName = f"{tasks[trainBeginTask - 1]}_{type(model.nnet).__name__}.tar"
        else:
            fileName = f"{tasks[trainBeginTask - 1]}_{type(model.nnet).__name__}_blip.tar"
        model.load_checkpoint(folder= './model_expThr_0.7', filename= fileName)

    for (task, param) in enumerate(tasks):
        if task < trainBeginTask:
            continue
        
        model.prepareNextTask(task, expandThreshold= expandThreshold[task])
        model.setCurrTask(task)
        log.info(f'task {task}: {param}')
        game = OthelloGameWrapper(param)
        coach = Coach(game, model, args)
        trainExamples = coach.learn()
        model.save_checkpoint(folder= './model', filename= f'{param}_{type(model.nnet).__name__}.tar')

        if isinstance(model, PQNetWrapper):
            model.applyBlip(f"cuda:{GPU_NUM}", trainExamples)
            freezeResult = used_capacity(model.nnet, 20)
            for (name, info) in freezeResult[1]:
                log.info(f"{name}: {info}")
            log.info(f'used capacity: {freezeResult[0]}')
            model.save_checkpoint(folder= './model', filename= f'{param}_{type(model.nnet).__name__}_blip.tar')

if __name__ == "__main__":
    main()
