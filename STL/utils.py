import torch
from torch.legacy.nn import Module
from torch.legacy.nn import Sequential
from torch.legacy.nn import SpatialSubtractiveNormalization
from torch.legacy.nn import SpatialDivisiveNormalization
import numpy as np
t = np.array([[ 0.0529,  0.1197,  0.1954,  0.2301,  0.1954,  0.1197,  0.0529],
    [0.1197,  0.2709,  0.4421,  0.5205,  0.4421,  0.2709,  0.1197],
    [0.1954,  0.4421,  0.7214,  0.8494,  0.7214,  0.4421,  0.1954],
    [0.2301,  0.5205,  0.8494,  1.0000,  0.8494,  0.5205,  0.2301],
    [0.1954,  0.4421,  0.7214,  0.8494,  0.7214,  0.4421,  0.1954],
    [0.1197,  0.2709,  0.4421,  0.5205,  0.4421,  0.2709,  0.1197],
    [0.0529,  0.1197,  0.1954,  0.2301,  0.1954,  0.1197,  0.0529]])
t = torch.from_numpy(t)
def SpatialContrast(nInputPlane=3):
    return SpatialContrastiveNormalization(nInputPlane=nInputPlane,kernel=t)
class SpatialContrastiveNormalization(Module):

    def __init__(self, nInputPlane=1, kernel=None, threshold=1e-4, thresval=1e-4):
        super(SpatialContrastiveNormalization, self).__init__()

        # get args
        self.nInputPlane = nInputPlane
        if kernel is None:
            self.kernel = torch.Tensor(9, 9).fill_(1)
        else:
            self.kernel = kernel
        self.threshold = threshold
        self.thresval = thresval or threshold
        kdim = self.kernel.ndimension()

        # check args
        if kdim != 2 and kdim != 1:
            raise ValueError('SpatialContrastiveNormalization averaging kernel must be 2D or 1D')

        if self.kernel.size(0) % 2 == 0 or (kdim == 2 and (self.kernel.size(1) % 2) == 0):
            raise ValueError('SpatialContrastiveNormalization averaging kernel must have ODD dimensions')

        # instantiate sub+div normalization
        self.normalizer = Sequential()
        self.normalizer.add(SpatialSubtractiveNormalization(self.nInputPlane, self.kernel))
        self.normalizer.add(SpatialDivisiveNormalization(self.nInputPlane, self.kernel,
                                                         self.threshold, self.thresval))

    def updateOutput(self, input):
        self.output = self.normalizer.forward(input)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = self.normalizer.backward(input, gradOutput)
        return self.gradInput