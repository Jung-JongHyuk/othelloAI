from train.network.quantizedLayer import Conv2d_Q, Linear_Q
import torch
import torch.nn as nn
import torch.nn.functional as F
import bisect

from torch.nn.modules.conv import Conv2d

class ExtendableLayer(nn.Linear):
    def __init__(self, layerType, *args, **kwargs):
        super(ExtendableLayer, self).__init__(1,1)
        self.register_buffer("insertedTaskPoint", torch.Tensor([0] + [100] * 100))
        self.register_buffer("layerSize", torch.Tensor([1]))
        self.layerType = layerType
        self.args = args
        self.kwargs = kwargs
        self.layers = nn.ModuleList([layerType(*args, **kwargs)])
    
    def forward(self, input, task):
        layerIdx = bisect.bisect_right(self.insertedTaskPoint.tolist(), task)
        return self.layers[layerIdx - 1](input)
    
    def fitLayerSize(self):
        currLayerSize = len(self.layers)
        targetLayerSize = int(self.layerSize.tolist()[0])
        for _ in range(targetLayerSize - currLayerSize):
            newLayer = self.layerType(*self.args, **self.kwargs)
            if torch.cuda.is_available():
                newLayer.cuda()
            if isinstance(self.layers[0], Conv2d_Q) or isinstance(self.layers[0], Linear_Q):
                newLayer.updateFisher = False
            self.layers.append(newLayer)
        if isinstance(self.layers[0], Conv2d_Q) or isinstance(self.layers[0], Linear_Q):
            self.layers[-1].updateFisher = True
    
    def extendLayer(self, task):
        newLayer = self.layerType(*self.args, **self.kwargs)
        if torch.cuda.is_available():
            newLayer.cuda()
        if isinstance(self.layers[-1], Conv2d_Q) or isinstance(self.layers[-1], Linear_Q):
            self.layers[-1].updateFisher = False
        self.layers.append(newLayer)
        self.layerSize = torch.Tensor([len(self.layers)])
        self.insertedTaskPoint[len(self.layers) - 1] = task

if __name__ == "__main__":
    net = ExtendableLayer(nn.Linear, 512, out_features= 10).cuda()
    net.extendLayer(1)
    print(net.state_dict())
    input = torch.randn(1, 512)
    print(net(input, 0))