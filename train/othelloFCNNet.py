import torch
import torch.nn as nn
import torch.nn.functional as F

class OthelloFCNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(OthelloFCNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, int(args.num_channels / 2), 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(int(args.num_channels / 2), int(args.num_channels / 2), 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(int(args.num_channels / 2), args.num_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)

        self.convBn1 = nn.BatchNorm2d(int(args.num_channels / 2))
        self.convBn2 = nn.BatchNorm2d(int(args.num_channels / 2))
        self.convBn3 = nn.BatchNorm2d(args.num_channels)
        self.convBn4 = nn.BatchNorm2d(args.num_channels)

        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(args.num_channels, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(args.num_channels, 1)
        self.fc4 = nn.Linear(int(args.num_channels / 4), 1)

        self.fcBn1 = nn.BatchNorm1d(1024)
        self.fcBn2 = nn.BatchNorm1d(512)

        self.conv5 = nn.Conv2d(args.num_channels, int(args.num_channels / 2), 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(int(args.num_channels / 2), int(args.num_channels / 4), 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(int(args.num_channels / 4), 1, 3, stride=1, padding=1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, s.shape[1], s.shape[2])                # batch_size x 1 x board_x x board_y
        s = F.relu(self.convBn1(self.conv1(s)))                          # batch_size x num_channels / 2 x board_x x board_y
        s = F.relu(self.convBn2(self.conv2(s)))                          # batch_size x num_channels / 2 x board_x x board_y
        s = F.relu(self.convBn3(self.conv3(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.convBn4(self.conv4(s)))                          # batch_size x num_channels x board_x x board_y

        glbpooled = nn.AdaptiveAvgPool2d(1)(s) # batch_size x num_channels x 1 x 1
        glbpooled = glbpooled.view(-1, self.args.num_channels) # batch_size x num_channels
        value = F.dropout(F.relu(self.fcBn1(self.fc1(glbpooled))), p=self.args.dropout, training=self.training) # batch_size x 1024
        value = F.dropout(F.relu(self.fcBn2(self.fc2(value))), p=self.args.dropout, training=self.training) # batch_size x 512
        value = self.fc3(value)

        pi = self.conv5(s) # batch_size x num_channels / 2 x board_x x board_y
        pi = self.conv6(pi) # batch_size x num_channels / 4 x board_x x board_y
        piPos = self.conv7(pi) # batch_size x 1 x board_x x board_y
        piPos = piPos.view(s.shape[0], -1) # batch_size x board_x * board_y
        piNull = nn.AdaptiveAvgPool2d(1)(pi) # batch_size x num_channels / 4 x 1 x 1
        piNull = piNull.view(s.shape[0], -1) # batch_size x num_channels / 4
        piNull = self.fc4(piNull) # batch_size x 1
        pi = torch.cat((piPos, piNull), 1) # batch_size x board_x * board_y + 1

        return F.log_softmax(pi, dim=1), torch.tanh(value)