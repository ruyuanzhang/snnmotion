# RNN for motion direction judgement, we call centsurrnet
import torch
from torch import nn
from torch.nn.functional import conv2d
import numpy as np
from CTRNN import CTRNN # import contious time recurrent neural network


class motionNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=4):
        super(motionNet, self).__init__()
        # save parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.state = None

        # ======= set different layers ==========
        # 1nd step, feed output of V1 filters to RNN
        self.rnn = CTRNN(input_size=input_size,
                           hidden_size=self.hidden_size, dt=8.3, tau=100)
        #self.rnn = nn.RNN(input_size=input_size,
        #                   hidden_size=self.hidden_size, num_layers=1)
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):  # inputs: (batch, seq_len)
        # 2nd step, calculate the gabor filter output
        Y, self.state = self.rnn(inputs)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, output_size)
        output = self.linear_layer(Y.view(-1, Y.shape[-1]))
        output = self.Softmax(output)
        return output, self.state
