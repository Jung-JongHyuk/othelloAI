import logging

# import coloredlogs

from train.Coach import Coach
from train.othelloGameWrapper import OthelloGameWrapper
from train.network.othelloNetWrapper import OthelloNetWrapper
from train.network.QNetWrapper import QNetWrapper
from train.utils import *

log = logging.getLogger(__name__)

# coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 2,
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

    game = OthelloGameWrapper(boardSize= (6,6), numOfBlock= 0)
    model = QNetWrapper(game)
    for boardSize in [(6,6), (8,8), (10,10)]:
        game = OthelloGameWrapper(boardSize= boardSize, numOfBlock= 0)
        coach = Coach(game, model, args)
        coach.learn()

if __name__ == "__main__":
    main()
