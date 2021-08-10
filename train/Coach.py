import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from train.othelloGameWrapper import OthelloGameWrapper
from train.network.PQFCNNet import PQFCNNet

import torch
import torch.nn as nn
import numpy as np
import copy
from train.network.quantizedLayer import Linear_Q, Conv2d_Q
from tqdm import tqdm

from .Arena import Arena
from .MCTS import MCTS
from game import Game
from board import Board
from player.randomPlayer import RandomPlayer
from player.aiPlayer import AIPlayer
from trainTasks import tasks

logFilePostfix = "PQFCN"

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
while log.hasHandlers():
    log.removeHandler(log.handlers[0])
streamHandler = logging.StreamHandler()
fileHandler = logging.FileHandler(f'trainLog_{logFilePostfix}.log', 'w')
log.propagate = False
log.addHandler(streamHandler)
log.addHandler(fileHandler)

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

        if args.load_model:
            self.loadTrainExamples()
            self.nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        startIter = self.args.resume_iter if self.args.load_model else 1

        for iter in range(startIter, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{iter} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or iter > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            # self.saveTrainExamples(iter - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=f'temp_{type(self.nnet.nnet).__name__}.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename=f'temp_{type(self.nnet.nnet).__name__}.pth.tar')

            pmcts = MCTS(self.game, self.pnet, self.args)
            # self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='curr.pth.tar')
            self.nnet.train(trainExamples)

            # for (i,j) in zip(self.nnet.nnet.modules(), self.pnet.nnet.modules()):
            #     # print(i._get_name())
            #     # if isinstance(i, Linear_Q) or isinstance(i, Conv2d_Q):
            #     #     i.weight.data.copy_(j.weight.data)
            #     #     i.bias.data.copy_(j.bias.data)
            #     #     # print("nnet weight: ", i.weight.data)
            #     #     # print("pnet weight: ", j.weight.data)
            #     #     # print(torch.sum(i.weight.data - j.weight.data))
            #     if isinstance(i, nn.BatchNorm2d):
            #         i.weight.data.copy_(j.weight.data)
            #         i.bias.data.copy_(j.bias.data)

            nmcts = MCTS(self.game, self.nnet, self.args)

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=f'curr_{type(self.nnet.nnet).__name__}.pth.tar')
            currLearningTask = self.nnet.currTask
            log.info('PITTING AGAINST RANDOM AGENT')
            for i in range(currLearningTask + 1):
                self.nnet.setCurrTask(i)
                wins = self.playWithRandomAgent(tasks[i], 100)
                log.info(f'{tasks[i]} : NEW/RANDOM WINS : {wins[0]} / {wins[1]} ; DRAWS : {100 - wins[0] - wins[1]}')
            self.nnet.setCurrTask(currLearningTask)

            self.pnet.setCurrTask(currLearningTask)
            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename=f'temp_{type(self.nnet.nnet).__name__}.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=f'best_{type(self.nnet.nnet).__name__}.pth.tar')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(iter))
        return trainExamples
    
    def playWithRandomAgent(self, param, iterCount):
        wins = [0,0]
        for _ in tqdm(range(int(iterCount / 2)), desc=f"Play with random(1), {param}"):
            board = Board(**param)
            game = Game(board, (AIPlayer(game= OthelloGameWrapper(param), agent= self.nnet), RandomPlayer()))
            winner = game.play(printBoard= False)
            if winner != None:
                wins[winner] = wins[winner] + 1
        for _ in tqdm(range(int(iterCount / 2)), desc=f"Play with random(2), {param}"):
            board = Board(**param)
            game = Game(board, (RandomPlayer(), AIPlayer(game= OthelloGameWrapper(param), agent= self.nnet)))
            winner = game.play(printBoard= False)
            if winner != None:
                wins[1 - winner] = wins[1 - winner] + 1
        return wins

    def getCheckpointFile(self, iteration):
        param = tasks[self.nnet.currTask]
        return f'{type(self.nnet).__name__}_{param}_checkpoint_{iteration}.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        # modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        # examplesFile = modelFile + ".examples"
        examplesFile = os.path.join(self.args.load_folder_file[0], self.getCheckpointFile(self.args.resume_iter - 1) + ".examples")
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
