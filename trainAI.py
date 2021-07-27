import logging
import math
from pickle import Pickler, Unpickler
import gc

from train.Coach import Coach
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
fileHandler = logging.FileHandler('fisherLog.log', 'w')
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
    'load_folder_file': ('/models/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    # log.info('Loading %s...', OthelloGameWrapper.__name__)
    # game = OthelloGameWrapper(boardSize= None, numOfBlock= None)
    # log.info('Loading %s...', OthelloNetWrapper.__name__)
    # model = OthelloNetWrapper(game)

    # if args.load_model:
    #     log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
    #     model.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    # else:
    #     log.warning('Not loading a checkpoint!')

    # log.info('Loading the Coach...')
    # coach = Coach(game, model, args)

    # if args.load_model:
    #     log.info("Loading 'trainExamples' from file...")
    #     c.loadTrainExamples()

    # log.info('Starting the learning process 🎉')
    # coach.learn()
    GPU_NUM = 1
    device = torch.device(f"cuda:{GPU_NUM}") if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)
    game = OthelloGameWrapper(boardSize= (6,6), blockPosType= "none")
    ModelType = OthelloNetWrapper
    model = ModelType(game)
    trainBeginTask = 0

    # model.load_checkpoint(folder='temp', filename='PQNetWrapper_(6, 6)_none_checkpoint_24.pth.tar')
    # trainExamples = []
    # with open(f'./temp/PQNetWrapper_(6, 6)_none_checkpoint_23.pth.tar.examples', "rb") as f:
    #     trainExamplesHistory = Unpickler(f).load()
    #     for e in trainExamplesHistory:
    #         trainExamples.extend(e)

    # estimate_fisher(0, f'cuda:{GPU_NUM}', model, trainExamples)
    # for m in model.nnet.modules():
    #     if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
    #         # update bits according to information gain
    #         m.update_bits(task=0, C=0.5/math.log(2))
    #         # with torch.no_grad():
    #         #     m.bit_alloc_w = torch.full(m.bit_alloc_w.shape, 20).cuda()
    #         #     m.bit_alloc_b = torch.full(m.bit_alloc_b.shape, 20).cuda()
    #         # do quantization
    #         m.sync_weight()
    #         # update Fisher in the buffer
    #         m.update_fisher(task=0)
    # freezeResult = used_capacity(model.nnet, 20)
    # for (name, info) in freezeResult[1]:
    #     log.info(f"{name}: {info}")
    # log.info(f'used capacity: {freezeResult[0]}')

    for (task, param) in enumerate(tasks):
        if task < trainBeginTask:
            continue
        if isinstance(model, PQNetWrapper):
            model.prepareNextTask(task)

        model.setCurrTask(task)
        boardSize = param["boardSize"]
        blockPosType = param["blockPosType"]
        log.info(f'task {task}: {boardSize}, {blockPosType}')
        game = OthelloGameWrapper(boardSize= boardSize, blockPosType= blockPosType)
        coach = Coach(game, model, args)
        coach.learn()
        model.save_checkpoint(folder= './model', filename= f'{boardSize}_{blockPosType}_{type(model.nnet).__name__}.tar')

        if isinstance(model, PQNetWrapper):
            trainExamples = []
            with open(f'./temp/PQNetWrapper_{boardSize}_{blockPosType}_checkpoint_{args.numIters - 1}.pth.tar.examples', "rb") as f:
                trainExamplesHistory = Unpickler(f).load()
                for e in trainExamplesHistory:
                    trainExamples.extend(e)
            estimate_fisher(task, f'cuda:{GPU_NUM}', model, trainExamples)
            for m in model.nnet.modules():
                if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
                    # update bits according to information gain
                    m.update_bits(task=task, C=0.5/math.log(2))
                    # do quantization
                    m.sync_weight()
                    # update Fisher in the buffer
                    m.update_fisher(task=task)
            freezeResult = used_capacity(model.nnet, 20)
            for (name, info) in freezeResult[1]:
                log.info(f"{name}: {info}")
            log.info(f'used capacity: {freezeResult[0]}')

if __name__ == "__main__":
    main()
