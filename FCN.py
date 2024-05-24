import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(3407)
class Actor(nn.Module):
    def __init__(self, n_act=9):
        super(Actor, self).__init__()
        self.data = []
        self.n_act = n_act
        nf = 64
        self.nf = nf
        self.conv = nn.Sequential(
            (nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=True)),
            nn.ReLU(),
            (nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(2, 2), dilation=2, bias=True)),
            nn.ReLU(),
            (nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(3, 3), dilation=3, bias=True)),
            nn.ReLU(),
            (nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(4, 4), dilation=4, bias=True)),
            nn.ReLU(),
        )

        self.diconv1_p1 = (nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(3, 3), dilation=3,
                                   bias=True))
        self.diconv2_p1 = (nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(2, 2), dilation=2,
                                   bias=True))
        self.outc_piRGB = nn.Conv2d(in_channels=64, out_channels=self.n_act*3, kernel_size=3, stride=1, padding=(1, 1), bias=True)

        self.diconv1_p2 = (
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(3, 3), dilation=3,
                      bias=True))
        self.diconv2_p2 = (
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(2, 2), dilation=2,
                      bias=True))
        self.outc_meanRGB = nn.Conv2d(in_channels=64, out_channels=self.n_act*3, kernel_size=3, stride=1, padding=(1, 1),
                                 bias=True)
        self.outc_logstdRGB = nn.Parameter(torch.zeros(1, self.n_act*3), requires_grad=True)


        self.diconv1_v = (nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(3, 3), dilation=3,
                                   bias=True))
        self.diconv2_v = (nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(2, 2), dilation=2,
                                   bias=True))
        self.value = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        # ----------------------------------------------------------
        self.conv7_Wz1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_Uz1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_Wr1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_Ur1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_W1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_U1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        #
        self.conv7_Wz2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_Uz2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_Wr2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_Ur2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_W2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_U2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)

        kernel = torch.zeros((1, 1, 33, 33))
        kernel[:, :, 16, 16] = 1
        self.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.bias = nn.Parameter(data=torch.zeros(1), requires_grad=False)

    # RMC
    def conv_smooth(self, x):
        x = F.conv2d(x, self.weight, self.bias, stride=1, padding=16)
        return x

    def parse_p(self, u_out):
        p = torch.mean(u_out.view(u_out.shape[0], u_out.shape[1], -1), dim=2)
        return p

    def forward(self, x):
        B, C, H, W = x[:, 0:3, :, :].size()
        x_in = copy.deepcopy(x[:, 0:3, :, :])
        ht1 = copy.deepcopy(x[:, 3:64 + 3, :, :])
        ht2 = copy.deepcopy(x[:, 64 + 3:64 + 3 + 64, :, :])

        conv = self.conv(x_in)

        p1 = self.diconv1_p1(conv)
        p1 = F.relu(p1)
        p1 = self.diconv2_p1(p1)
        p1 = F.relu(p1)
        GRU_in1 = p1
        z_t = torch.sigmoid(self.conv7_Wz1(GRU_in1) + self.conv7_Uz1(ht1))
        r_t = torch.sigmoid(self.conv7_Wr1(GRU_in1) + self.conv7_Ur1(ht1))
        h_title_t = torch.tanh(self.conv7_W1(GRU_in1) + self.conv7_U1(r_t * ht1))
        h_t1 = (1 - z_t) * ht1 + z_t * h_title_t

        policyRGB = self.outc_piRGB(h_t1).reshape(B*C, self.n_act, H, W)
        policy = F.softmax(policyRGB, 1)

        p2 = self.diconv1_p2(conv)
        p2 = F.relu(p2)
        p2 = self.diconv2_p2(p2)
        p2 = F.relu(p2)
        GRU_in2 = p2
        z_t = torch.sigmoid(self.conv7_Wz2(GRU_in2) + self.conv7_Uz2(ht2))
        r_t = torch.sigmoid(self.conv7_Wr2(GRU_in2) + self.conv7_Ur2(ht2))
        h_title_t = torch.tanh(self.conv7_W2(GRU_in2) + self.conv7_U2(r_t * ht2))
        h_t2 = (1 - z_t) * ht2 + z_t * h_title_t
        meanRGB = self.parse_p(self.outc_meanRGB(h_t2)).reshape(B*3, self.n_act)
        logstdRGB = self.outc_logstdRGB.expand([B, self.n_act*3]).reshape(B*3, self.n_act, 1, 1)

        v = self.diconv1_v(conv)
        v = F.relu(v)
        v = self.diconv2_v(v)
        v = F.relu(v)
        value = self.value(v).reshape(B*3, 1, H, W)
        return policy, value, meanRGB, logstdRGB, h_t1, h_t2

# test
# import time
# from thop import profile
# if __name__ == '__main__':
#     actor = Actor().cuda()
#     ins = torch.randn(1, 3+64, 512, 512).cuda()
#     policy, value = actor(ins)
#     policy, value = actor(ins)
#     policy, value = actor(ins)
#     t1 = time.time()
#     # policy, value, meanRGB, logstdRGB, h_t1, h_t2 = actor(ins)
#
#
#     policy, value = actor(ins)
#
#
#     t2 = time.time()
#     print(t2-t1)
#
#     flops, params = profile(actor, inputs=(ins,))
#     print(flops / (1000 ** 3), params / (1000 ** 2))
#
#     # print(policy.shape)
#     # print(value.shape)
#     # print(meanRGB.shape)
#     # print(logstdRGB.shape)
#     # print(h_t1.shape)
#     # print(h_t2.shape)
