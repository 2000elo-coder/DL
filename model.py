import torch
import torch.nn.functional as F


class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        # --- UPGRADED: Width increased to 60 as requested --- #
        width = 60

        self.L1 = torch.nn.Linear(2, width)
        self.L2 = torch.nn.Linear(width, width)
        self.L3 = torch.nn.Linear(width, width)
        self.L4 = torch.nn.Linear(width, width)
        self.L5 = torch.nn.Linear(width, width)
        self.L6 = torch.nn.Linear(width, width)
        self.L7 = torch.nn.Linear(width, 1)

    def forward(self, x, y):
        inputs = torch.cat([x, y], axis=1)
        x1 = F.silu(self.L1(inputs))
        x2 = F.silu(self.L2(x1))
        x3 = F.silu(self.L3(x2))
        x4 = F.silu(self.L4(x3))
        x5 = F.silu(self.L5(x4))
        x6 = F.silu(self.L6(x5))
        x7 = self.L7(x6)
        return x7


class VonKarmanPINN(torch.nn.Module):
    def __init__(self):
        super(VonKarmanPINN, self).__init__()
        self.net_u = NN()
        self.net_v = NN()

    def forward(self, x, y):
        u_pred = self.net_u(x, y)
        v_pred = self.net_v(x, y)
        return u_pred, v_pred


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)