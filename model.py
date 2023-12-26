import torch.nn as nn
import torch
from torch.nn.utils import weight_norm


class Decoder(nn.Module):
    def __init__(
        self,
        args,
        dropout_prob=0.1,
    ):
        super(Decoder, self).__init__()
        self.fc1 = weight_norm(nn.Linear(3, 512))
        self.fc2 = weight_norm(nn.Linear(512, 512))
        self.fc3 = weight_norm(nn.Linear(512, 512))
        self.fc4 = weight_norm(nn.Linear(512, 509))
        self.fc5 = weight_norm(nn.Linear(512, 512))
        self.fc6 = weight_norm(nn.Linear(512, 512))
        self.fc7 = weight_norm(nn.Linear(512, 512))
        self.fc8 = nn.Linear(512, 1)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.tanh = nn.Tanh()

    # input: N x 3
    def forward(self, x_input):
        x = self.dropout(self.prelu(self.fc1(x_input)))
        x = self.dropout(self.prelu(self.fc2(x)))
        x = self.dropout(self.prelu(self.fc3(x)))
        x = self.dropout(self.prelu(self.fc4(x)))
        x = torch.cat((x, x_input), dim=1)
        x = self.dropout(self.prelu(self.fc5(x)))
        x = self.dropout(self.prelu(self.fc6(x)))
        x = self.dropout(self.prelu(self.fc7(x)))
        x = self.fc8(x)
        x = self.tanh(x)

        return x
