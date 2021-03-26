import torch.nn as nn
import torch


class AR(nn.Module):

    def __init__(self, window):  # window 时间步

        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        return x


"""
init:
    self.sgsf = Single_Global_SelfAttn_Module()
    self.slsf = Single_Local_SelfAttn_Module()
    self.active_func = nn.Tanh()
    self.W_output1 = nn.Linear(2 * self.n_kernels, 1)
    self.ar = AR(window=self.window)
    self.dropout = nn.Dropout(p=self.drop_prob)

forward:
    sgsf_output, *_ = self.sgsf(x)
    slsf_output, *_ = self.slsf(x)
    sf_output = torch.cat((sgsf_output, slsf_output), 2)
    sf_output = self.dropout(sf_output)
    sf_output = self.W_output1(sf_output)

    sf_output = torch.transpose(sf_output, 1, 2) 
    ar_output = self.ar(x)
    
    output = sf_output + ar_output

    return output

"""