import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class PreProcessorFold(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.fold = Conv(input_channel, output_channel)

    def forward(self, x):
        return self.fold(x)


class PreProcessorUnfold(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.unfold = Conv(input_channel, output_channel)

    def forward(self, x):
        return self.unfold(x)


class PreProcessor(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.fold = PreProcessorFold(input_channel, output_channel)
        self.unfold = PreProcessorUnfold(output_channel, input_channel)

    def forward(self, original):
        folded = self.fold(original)
        unfolded = self.unfold(folded)
        return unfolded
