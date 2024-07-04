from torch import nn


class Adapter(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.fold = nn.Conv2d(input_channel, 16, 1)
        # if self.training:
        #     self.unfold = nn.Conv2d(16, input_channel, 1)

    def forward(self, x):
        # if self.training:
        #     folded = self.fold(x)
        #     unfolded = self.unfold(folded)
        #     raise ValueError
        #     return unfolded
        # else:
        return self.fold(x)


class MultiAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter_yrcc1 = Adapter(3)
        self.adapter_yrcc2 = Adapter(3)
        self.adapter_yrccms = Adapter(4)
        self.adapter_albert = Adapter(3)

    def forward(self, x):
        if self.dataset == 0:
            return self.adapter_yrcc1(x)
        elif self.dataset == 1:
            return self.adapter_yrcc2(x)
        elif self.dataset == 2:
            return self.adapter_yrccms(x)
        elif self.dataset == 3:
            return self.adapter_albert(x)
        else:
            raise NotImplementedError
