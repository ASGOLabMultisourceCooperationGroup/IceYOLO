from torch import nn


class Adapter(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.fold = nn.Conv2d(input_channel, 16, 1)
        if self.training:
            self.unfold = nn.Conv2d(16, input_channel, 1)

    def forward(self, x):
        if self.training:
            folded = self.fold(x)
            unfolded = self.unfold(folded)
            return unfolded
        else:
            return self.fold(x)
