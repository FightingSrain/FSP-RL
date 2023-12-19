import numpy as np
import sys
import cv2
import copy
from utils import *
class State():
    def __init__(self, size, move_range):
        self.image = np.zeros(size, dtype=np.float32)
        self.move_range = move_range

    def reset(self, x):
        self.image = copy.deepcopy(x)
        self.pre_img = copy.deepcopy(self.image)
        self.tensor = copy.deepcopy(self.image)

    def hybrid_act(self, num, batch, par):
        k = 0.3
        b = [0., 0.3, 0.6, 0.9]
        return (b[num - 3] + k * sigmoid(par[batch][num])).astype(np.float32)

    def step(self, act, par):
        B, H1, W1 = act.shape
        _, L = par.shape
        act = np.expand_dims(act, 1).reshape(B//3, 3, H1, W1)
        par = par.reshape(B//3, 3, 9)

        pixel1 = np.zeros(self.image.shape, self.image.dtype)
        pixel2 = np.zeros(self.image.shape, self.image.dtype)

        gaussian = np.zeros(self.image.shape, self.image.dtype)
        gaussian2 = np.zeros(self.image.shape, self.image.dtype)
        bilateral = np.zeros(self.image.shape, self.image.dtype)
        bilateral2 = np.zeros(self.image.shape, self.image.dtype)
        median = np.zeros(self.image.shape, self.image.dtype)
        box = np.zeros(self.image.shape, self.image.dtype)

        for x in range(0, 3):
            b, c, h, w = self.image.shape
            for i in range(0, b):
                if np.sum(act[i, x] == self.move_range - 2) > 0:
                    pixel1[i, x] = np.expand_dims(self.image[i, x] + 0.5*sigmoid(par[i, x][1]), 0)

                if np.sum(act[i, x] == self.move_range - 1) > 0:
                    pixel2[i, x] = np.expand_dims(self.image[i, x] - 0.5*sigmoid(par[i, x][2]), 0)

                if np.sum(act[i, x] == self.move_range) > 0:
                    gaussian[i, x] = np.expand_dims(cv2.GaussianBlur(self.image[i, x].squeeze().astype(np.float32), ksize=(5, 5),
                                                                  sigmaX=sigmoid(par[i, x][3])), 0)  # (0,1)
                if np.sum(act[i, x] == self.move_range + 1) > 0:
                    bilateral[i, x] = np.expand_dims(
                        cv2.bilateralFilter(self.image[i, x].squeeze().astype(np.float32), d=5,
                                            sigmaColor=sigmoid(par[i, x][4])*0.5, sigmaSpace=5), 0)
                if np.sum(act[i, x] == self.move_range + 2) > 0:
                    median[i, x] = np.expand_dims(cv2.medianBlur(self.image[i, x].squeeze().astype(np.float32), ksize=5), 0)  # 5

                if np.sum(act[i, x] == self.move_range + 3) > 0:
                    gaussian2[i, x] = np.expand_dims(cv2.GaussianBlur(self.image[i, x].squeeze().astype(np.float32), ksize=(5, 5),
                                                                   sigmaX=sigmoid(par[i, x][6])+1), 0)  # (1,2)
                if np.sum(act[i, x] == self.move_range + 4) > 0:
                    bilateral2[i, x] = np.expand_dims(
                        cv2.bilateralFilter(self.image[i, x].squeeze().astype(np.float32), d=5,
                                            sigmaColor=sigmoid(par[i, x][7])*0.5 + 1, sigmaSpace=5), 0)
                if np.sum(act[i, x] == self.move_range + 5) > 0:
                    box[i, x] = np.expand_dims(
                        cv2.boxFilter(self.image[i, x].squeeze().astype(np.float32), ddepth=-1, ksize=(5, 5)), 0)


            self.image[:, x] = np.where(act[:, x, :, :] == self.move_range - 2, pixel1[:, x], self.image[:, x])
            self.image[:, x] = np.where(act[:, x, :, :] == self.move_range - 1, pixel2[:, x], self.image[:, x])
            self.image[:, x] = np.where(act[:, x, :, :] == self.move_ransge, gaussian[:, x], self.image[:, x])
            self.image[:, x] = np.where(act[:, x, :, :] == self.move_range + 1, bilateral[:, x], self.image[:, x])
            self.image[:, x] = np.where(act[:, x, :, :] == self.move_range + 2, median[:, x], self.image[:, x])
            self.image[:, x] = np.where(act[:, x, :, :] == self.move_range + 3, gaussian2[:, x], self.image[:, x])
            self.image[:, x] = np.where(act[:, x, :, :] == self.move_range + 4, bilateral2[:, x], self.image[:, x])
            self.image[:, x] = np.where(act[:, x, :, :] == self.move_range + 5, box[:, x], self.image[:, x])

        self.image = np.clip(self.image, a_min=0., a_max=1.)
        # self.image = np.concatenate([self.image, h_t1, h_t2], 1)
        self.tensor[:, :self.image.shape[1], :, :] = self.image


