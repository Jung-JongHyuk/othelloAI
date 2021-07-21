import torch
import torch.nn as nn
import torch.nn.functional as F
import bisect

class ExtendableLayer(nn.Module):
    def __init__(self, layerType, *args, **kwargs):
        super(ExtendableLayer, self).__init__()
        self.layerType = layerType
        self.args = args
        self.layers = nn.ModuleList([layerType(*args, **kwargs)])
        self.insertedTaskPoint = [0]
    
    def forward(self, input, task):
        layerIdx = bisect.bisect_right(self.insertedTaskPoint, task)
        outputs = [layer(input) for layer in self.layers[:layerIdx]]
        return sum(outputs)
    
    def extendLayer(self, task):
        self.layers.append(self.layerType(*self.args))
        self.insertedTaskPoint.append(task)

if __name__ == "__main__":
    net = ExtendableLayer(nn.Linear, 512, out_features= 10)
    input = torch.randn(1, 512)
    print(net(input, 0))