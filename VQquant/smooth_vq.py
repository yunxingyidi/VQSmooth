import torch
from torch import nn
import numpy as np

import math
import time
import json
import transformers

from VQquant.vector_quant import vq_quantize, VectorQuantizer

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def quad_loss(w_q, G, v, offset):
    # Quadratic loss: 1/2 wGw^T
    loss = 0.5 * (w_q.mm(G) * w_q).sum()
    # Add linear term and offset
    loss += (v * w_q).sum()
    loss += offset
    return loss


def quad_loss_2(W, Q, G):
    Werr = W - Q
    return (Werr.mm(G) * Werr).sum()

class SmoothVQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        # W = layer.weight.data.clone()
        # W = layer.weight.data
        # if isinstance(self.layer, nn.Conv2d):
        #     W = W.flatten(1)
        # if isinstance(self.layer, transformers.Conv1D):
        #     W = W.t()
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.quantizers = None
        # self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    # def add_batch(self, inp, out):
    #     if len(inp.shape) == 2:
    #         inp = inp.unsqueeze(0)
    #     tmp = inp.shape[0]
    #     if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D) or isinstance(self.layer, nn.Module):
    #         if len(inp.shape) == 3:
    #             inp = inp.reshape((-1, inp.shape[-1]))
    #         inp = inp.t()
    #     if isinstance(self.layer, nn.Conv2d):
    #         unfold = nn.Unfold(
    #             self.layer.kernel_size,
    #             dilation=self.layer.dilation,
    #             padding=self.layer.padding,
    #             stride=self.layer.stride,
    #         )
    #         inp = unfold(inp)
    #         inp = inp.permute([1, 0, 2])
    #         inp = inp.flatten(1)
    #     self.H *= self.nsamples / (self.nsamples + tmp)
    #     self.nsamples += tmp
    #     inp = math.sqrt(2 / self.nsamples) * inp.float()
    #     self.H += inp.matmul(inp.t())

    def fasterquant(
            self,
            use_vq=True,
            name="layer",
            fake_quant=False
    ):
        # W = self.layer.weight.data.clone()
        W = self.layer.weight.data
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        self.quantizer.get_centroids(W, weight=True)

        sub_vector = 1

        if use_vq:
            sub_vector = self.quantizer.sub_vector
            self.assignments = []

        # reshape back
        # optional M-step
        # if include_m_step:
        #     Q = self.lut_m_step(Q, self.quantizer)
        if fake_quant:
            Q, assmt = vq_quantize(W, self.quantizer, fake_quant=fake_quant)
            if isinstance(self.layer, transformers.Conv1D):
                Q = Q.t()
            # max_err = (W.cpu() - Q.cpu()).abs().mean().item()
            # print(max_err)
            return Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
            # self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        else:
            assmt = vq_quantize(W, self.quantizer, fake_quant=fake_quant)
            if use_vq:
                self.assignments.append(assmt)
            centroids = self.quantizer.all_centroids[-1]
            return assmt, centroids.permute(1, 0, 2).reshape(centroids.shape[1], -1)

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()