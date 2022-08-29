import numpy as np
import os
import torch.nn

class evaluate(object):
    def __init__(self):
        pass

    def load_all(self, path):
        data = np.load(path)
        return data

    def loss_mse(self, pred, gt):
        diff = pred - gt
        loss = np.sqrt(np.sum(diff ** 2))
        return loss

    def loss_mse_cloud(self, pred, gt, mask):
        diff = pred - gt
        loss = np.sqrt(np.sum(diff ** 2 * mask))
        return loss