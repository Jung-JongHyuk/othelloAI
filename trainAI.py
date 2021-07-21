import logging
import math
from pickle import Pickler, Unpickler
# import coloredlogs

from train.Coach import Coach
from train.othelloGameWrapper import OthelloGameWrapper
from train.network.othelloNetWrapper import OthelloNetWrapper
from train.network.QNetWrapper import QNetWrapper
from train.utils import *
from train.blip_utils import *

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
    'numIters': 50,
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

    import gc

    gc.collect()

    torch.cuda.empty_cache()

    game = OthelloGameWrapper(boardSize= (6,6), blockPosType= 'none')
    model = QNetWrapper(game)
    model.load_checkpoint(folder='./temp/', filename='QNetWrapper_(6, 6)_none_checkpoint_29.pth.tar')
    # trainExamples = []
    # with open(f'./temp/QNetWrapper_(6, 6)_none_checkpoint_28.pth.tar.examples', "rb") as f:
    #     trainExamplesHistory = Unpickler(f).load()
    #     for e in trainExamplesHistory:
    #         trainExamples.extend(e)

    # estimate_fisher(0, 'cuda:0', model, trainExamples)
    # for m in model.nnet.modules():
    #     if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
    #         # update bits according to information gain
    #         m.update_bits(task=0, C=0.5/math.log(2))
    #         # do quantization
    #         m.sync_weight()
    #         # update Fisher in the buffer
    #         m.update_fisher(task=0)
    # print(used_capacity(model.nnet, 20))
    
    # for (task, (boardSize, blockPosType)) in enumerate([((6,6), 'none'), ((6,6), 'x-cross'), ((6,6), 'cross'), ((6,6), 'octagon')]):
    for (task, (boardSize, blockPosType)) in enumerate([((6,6), 'x-cross')]):
        log.info(f'task {task}: {boardSize}, {blockPosType}')
        game = OthelloGameWrapper(boardSize= boardSize, blockPosType= blockPosType)
        coach = Coach(game, model, args)
        coach.learn()
        model.save_checkpoint(folder= './model/', filename= f'{boardSize}_{blockPosType}_{type(model.nnet).__name__}.tar')
        if isinstance(model, QNetWrapper):
            trainExamples = []
            with open(f'./temp/checkpoint_{args.numIters - 1}.pth.tar.examples', "rb") as f:
                trainExamplesHistory = Unpickler(f).load()
                for e in trainExamplesHistory:
                    trainExamples.extend(e)
            estimate_fisher(task, 'cuda:0', model, trainExamples)
            for m in model.nnet.modules():
                if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
                    # update bits according to information gain
                    m.update_bits(task=task, C=0.5/math.log(2))
                    # do quantization
                    m.sync_weight()
                    # update Fisher in the buffer
                    m.update_fisher(task=task)
            freezeResult = used_capacity(model.nnet, 20)
            log.info(freezeResult[1])
            log.info(f'used capacity: f{freezeResult[0]}')

if __name__ == "__main__":
    main()
