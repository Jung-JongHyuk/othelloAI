import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizedLayer import Linear_Q, Conv2d_Q

class QFCNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(QFCNNet, self).__init__()
        self.valConv1 = Conv2d_Q(1, int(args.num_channels / 2), 3, stride=1, padding=1)
        self.valConv2 = Conv2d_Q(int(args.num_channels / 2), int(args.num_channels / 2), 3, stride=1, padding=1)
        self.valConv3 = Conv2d_Q(int(args.num_channels / 2), args.num_channels, 3, stride=1, padding=1)
        self.valConv4 = Conv2d_Q(args.num_channels, args.num_channels, 3, stride=1, padding=1)

        self.valConvBn1 = nn.BatchNorm2d(int(args.num_channels / 2))
        self.valConvBn2 = nn.BatchNorm2d(int(args.num_channels / 2))
        self.valConvBn3 = nn.BatchNorm2d(args.num_channels)
        self.valConvBn4 = nn.BatchNorm2d(args.num_channels)

        self.piConv1 = Conv2d_Q(1, int(args.num_channels / 2), 3, stride=1, padding=1)
        self.piConv2 = Conv2d_Q(int(args.num_channels / 2), int(args.num_channels / 2), 3, stride=1, padding=1)
        self.piConv3 = Conv2d_Q(int(args.num_channels / 2), args.num_channels, 3, stride=1, padding=1)
        self.piConv4 = Conv2d_Q(args.num_channels, args.num_channels, 3, stride=1, padding=1)

        self.piConvBn1 = nn.BatchNorm2d(int(args.num_channels / 2))
        self.piConvBn2 = nn.BatchNorm2d(int(args.num_channels / 2))
        self.piConvBn3 = nn.BatchNorm2d(args.num_channels)
        self.piConvBn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = Linear_Q(args.num_channels, 1024)
        self.fc2 = Linear_Q(1024, 512)
        self.fc3 = Linear_Q(args.num_channels, 1)
        self.fc4 = Linear_Q(int(args.num_channels / 4), 1)

        self.fcBn1 = nn.BatchNorm1d(1024)
        self.fcBn2 = nn.BatchNorm1d(512)

        self.conv5 = Conv2d_Q(args.num_channels, int(args.num_channels / 2), 3, stride=1, padding=1)
        self.conv6 = Conv2d_Q(int(args.num_channels / 2), int(args.num_channels / 4), 3, stride=1, padding=1)
        self.conv7 = Conv2d_Q(int(args.num_channels / 4), 1, 3, stride=1, padding=1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, s.shape[1], s.shape[2])                # batch_size x 1 x board_x x board_y
        s_val = F.relu(self.valConvBn1(self.valConv1(s)))                          # batch_size x num_channels / 2 x board_x x board_y
        s_val = F.relu(self.valConvBn2(self.valConv2(s_val)))                          # batch_size x num_channels / 2 x board_x x board_y
        s_val = F.relu(self.valConvBn3(self.valConv3(s_val)))                          # batch_size x num_channels x board_x x board_y
        s_val = F.relu(self.valConvBn4(self.valConv4(s_val)))                          # batch_size x num_channels x board_x x board_y

        s_pi = F.relu(self.piConvBn1(self.piConv1(s)))                          # batch_size x num_channels / 2 x board_x x board_y
        s_pi = F.relu(self.piConvBn2(self.piConv2(s_pi)))                          # batch_size x num_channels / 2 x board_x x board_y
        s_pi = F.relu(self.piConvBn3(self.piConv3(s_pi)))                          # batch_size x num_channels x board_x x board_y
        s_pi = F.relu(self.piConvBn4(self.piConv4(s_pi)))                          # batch_size x num_channels x board_x x board_y

        glbpooled = nn.AdaptiveAvgPool2d(1)(s_val) # batch_size x num_channels x 1 x 1
        glbpooled = glbpooled.view(-1, self.args.num_channels) # batch_size x num_channels
        value = F.dropout(F.relu(self.fcBn1(self.fc1(glbpooled))), p=self.args.dropout, training=self.training) # batch_size x 1024
        value = F.dropout(F.relu(self.fcBn2(self.fc2(value))), p=self.args.dropout, training=self.training) # batch_size x 512
        value = self.fc3(value)

        pi = self.conv5(s_pi) # batch_size x num_channels / 2 x board_x x board_y
        pi = self.conv6(pi) # batch_size x num_channels / 4 x board_x x board_y
        piPos = self.conv7(pi) # batch_size x 1 x board_x x board_y
        piPos = piPos.view(s.shape[0], -1) # batch_size x board_x * board_y
        piNull = nn.AdaptiveAvgPool2d(1)(pi) # batch_size x num_channels / 4 x 1 x 1
        piNull = piNull.view(s.shape[0], -1) # batch_size x num_channels / 4
        piNull = self.fc4(piNull) # batch_size x 1
        pi = torch.cat((piPos, piNull), 1) # batch_size x board_x * board_y + 1

        return F.log_softmax(pi, dim=1), torch.tanh(value)