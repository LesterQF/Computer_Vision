from torch import nn

class SE_Block(nn.Module):
    def __init__(self, ratio, in_channels) -> None:
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d((1, 1))
        self.model=nn.Sequential(
            nn.Linear(in_channels, in_channels//ratio),
            nn.ReLU(),
            nn.Linear(in_channels//ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self,x):
        b, c, h, w = x.size()
        y=self.gap(x).view(b,c)
        y = self.model(y).view(b,c,1,1)
        return x * y.expand_as(x)
