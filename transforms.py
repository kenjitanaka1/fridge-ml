import math
import numpy as np
import torch

class ConstantPad(torch.nn.Module):

    def __init__(self, shape, padding_mode='constant'):
        super().__init__()
        self.shape = shape
        self.padding_mode = padding_mode


    def forward(self, img):
        assert len(self.shape) == len(img.shape)
        pad_width = list((int(math.floor(abs(a-b)/2)), int(math.ceil(abs(a-b)/2))) for a,b in zip(self.shape, img.shape))
        return np.pad(img, pad_width, mode=self.padding_mode)

if __name__=='__main__':
    pad = ConstantPad((20,20,3), padding_mode='reflect')
    img = np.random.randint(10, size=(18,11,3))
    print(img[:,:,0])

    padded = pad(img)
    print(padded[:,:,0])