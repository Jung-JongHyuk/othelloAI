import torch
import torch.nn as nn
import torch.nn.functional as F
import bisect

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
        # outputs = [layer(input) for layer in self.layers[:layerIdx]]
        # return sum(outputs)
    
    def fitLayerSize(self):
        currLayerSize = len(self.layers)
        targetLayerSize = int(self.layerSize.tolist()[0])
        for _ in range(targetLayerSize - currLayerSize):
            newLayer = self.layerType(*self.args, **self.kwargs)
            if torch.cuda.is_available():
                newLayer.cuda()
            self.layers.append(newLayer)
    
    def extendLayer(self, task):
        newLayer = self.layerType(*self.args, **self.kwargs)
        if torch.cuda.is_available():
            newLayer.cuda()
        self.layers.append(newLayer)
        self.layerSize = torch.Tensor([len(self.layers)])
        self.insertedTaskPoint[len(self.layers) - 1] = task

if __name__ == "__main__":
    net = ExtendableLayer(nn.Linear, 512, out_features= 10).cuda()
    net.extendLayer(1)
    print(net.state_dict())
    input = torch.randn(1, 512)
    print(net(input, 0))