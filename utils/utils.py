import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import os

def save_checkpoint(model, epoch, logpath):
    state = {
            'net': model.state_dict(),
        }
    with tf.io.gfile.GFile(os.path.join(logpath, f'ckpt_{epoch}.t7'), 'wb') as f:
        torch.save(state, f)


def compute_per_class_acc(test_label, predicted_label, nclass):
    acc_per_class = []
    acc = np.sum(test_label == predicted_label) / len(test_label)
    for i in range((nclass)):
        idx = (test_label == i)
        if np.sum(idx) > 0:
            acc_per_class.append(np.sum(test_label[idx] == predicted_label[idx]) / np.sum(idx))
    # print(len(acc_per_class))
    return acc, sum(acc_per_class)/len(acc_per_class)

class View(nn.Module):
    def __init__(self, feats):
        super(View, self).__init__()
        self.feats = feats

    def forward(self, x):
        b = x.shape[0]
        pre_pooled =  x.view(b, -1, self.feats)
        pooled = pre_pooled.mean(1).unsqueeze(1)
        return torch.cat([pooled, pre_pooled], 1)

def file_to_list(filename):
    with open(filename, 'r') as f:
        things = f.readlines()
    output = []
    for a in things:
        output.append(a.strip())
    return output

class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S = 0.0
        self.best_acc_U = 0.0
        self.acc_list = []
        self.epoch_list = []
    def update(self, it, acc):
        self.acc_list += [acc]
        self.epoch_list += [it]
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_iter = it
    def update_gzsl(self, it, acc_u, acc_s, H):
        self.acc_list += [H]
        self.epoch_list += [it]
        if H > self.best_acc:
            self.best_acc = H
            self.best_iter = it
            self.best_acc_U = acc_u
            self.best_acc_S = acc_s