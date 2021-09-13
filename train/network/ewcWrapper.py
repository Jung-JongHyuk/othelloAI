import torch
import torch.optim as optim
import numpy as np
import random
import sys
import os
from tqdm import tqdm
from .NeuralNet import NeuralNet
from .othelloFCNNet import OthelloFCNNet
sys.path.append('../')
from train.utils import *
from train.ewc_utils import EWC

args = dotdict({
    'lr': 0.001,
    'ewcWeight': 1000,
    'oldExampleSampleSize': 200,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 256,
})

class EWCWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = OthelloFCNNet(game, args)
        self.boardSize = game.getBoardSize()
        self.actionSize = game.getActionSize()
        self.currTask = 0
        self.oldExamples = list()
        self.ewc = None
        if args.cuda:
            self.nnet.cuda()
    
    def setCurrTask(self, task):
        self.currTask = task
    
    def prepareNextTask(self, nextTask, expandThreshold):
        return

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards, self.currTask)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                if self.currTask != 0:
                    total_loss += args.ewcWeight * self.ewc.penalty(self.nnet)

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
    
    def updateEWC(self, oldExamples):
        self.oldExamples += random.sample(oldExamples, self.args.oldExamplesSampleSize)
        self.ewc = EWC(self.nnet, list(map(lambda example: example[0], self.oldExamples)))

    def predict(self, board):
        """
        board: np array with board
        """
        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        board = board.view(1, board.shape[0], board.shape[1])
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board, self.currTask)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
    
    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)
        print(f"saved in {filepath}")

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'], strict= False)
        for m in self.nnet.modules():
            if isinstance(m, ExtendableLayer):
                m.fitLayerSize()
        self.nnet.load_state_dict(checkpoint['state_dict'])