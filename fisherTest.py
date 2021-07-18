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

game = OthelloGameWrapper(boardSize= (6,6), numOfBlock= 0)
model = QNetWrapper(game)
model.load_checkpoint(folder='temp', filename='checkpoint_1.pth.tar')
trainExamples = []
with open(f'./temp/checkpoint_0.pth.tar.examples', "rb") as f:
    trainExamplesHistory = Unpickler(f).load()
    for e in trainExamplesHistory:
        trainExamples.extend(e)
estimate_fisher(0, 'cuda:0', model, trainExamples)
for m in model.nnet.modules():
    if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
        # update bits according to information gain
        m.update_bits(task=0, C=0.5/math.log(2))
        # do quantization
        m.sync_weight()
        # update Fisher in the buffer
        m.update_fisher(task=0)
print(used_capacity(model.nnet, 20))



    

