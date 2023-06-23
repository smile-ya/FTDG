import numpy as np
from .module import Module

class GlobalAveragePooling2D(Module):

    def __init__(self):
        Module.__init__(self)
        self.X = None

    def forward(self, X,*args,**kwargs):
        self.X = X
        N, H, W, C = X.shape
        self.Y = np.mean(X, axis=(1, 2))
        # print(self.Y.shape)
        return self.Y

    def backward(self, dout):
        N, C, H, W = self.X.shape
        # dX = np.zeros_like(self.X)
        dX = np.repeat(np.repeat(dout[:, :, np.newaxis, np.newaxis], H, axis=1), W, axis=2)
        dX /= (H * W)
        return dX
        # N, H, W, C = self.input_shape
        # grad_input = np.zeros(self.input_shape)
        # for i in range(N):
        #     for c in range(C):
        #         grad_input[i, :, :, c] = grad_output[i, c] / (H * W)
        # return grad_input

    # def lrp(self, R, *args,**kwargs):
    #     Zs = self.Y + 1e-16 * ((self.Y >= 0) * 2 - 1.)  # add weakdefault stabilizer to denominator
    #     if self.lrp_aware:
    #         return (self.Z * (R / Zs)[:, na, :]).sum(axis=2)
    #     else:
    #         Z = self.W[na, :, :] * self.X[:, :, na]  # localized preactivations
    #         return (Z * (R / Zs)[:, na, :]).sum(axis=2)

    def lrp(self, R, *args,**kwargs):

        # eps = 10 ** -10
        # z = self.Y + eps * np.sign(self.Y)  # adding small constant to avoid zero division
        # s = R / z
        # c = np.sum(s * self.Y, axis=(2, 3), keepdims=True)
        # # R = self.X * (s - c)
        # return R
        Z = np.zeros(self.X.shape)
        N, H, W, C = self.X.shape
        # X = self.X
        # out = self.forward(X)
        Z = np.tile(R / (N * C), (1, H, W,1))
        return Z

        # Z = np.zeros(self.input_shape)
        # N, C, H, W = self.input_shape
        # X = self.forward_input
        # out = self.forward(X)
        # mask = (X == out)
        # mask /= np.sum(mask, axis=(2, 3), keepdims=True) + self.epsilon
        # Z = np.tile(mask * R, (1, 1, H, W))
        # return Z
        # return np.reshape(R, self.X.shape)