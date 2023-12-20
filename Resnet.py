from torch import nn
from torch.nn import functional as F
class Residual(nn.Module):

    def __init__(self, in_channels, out_channels, use_1x1_conv=False, stride=1):
        super().__init__()
        if use_1x1_conv:
            self.res=nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride)
            )
        else:
            self.res=nn.Sequential()
        self.model=nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,x):
        ret=self.model(x)
        ret=ret+self.res(x)
        return F.relu(ret)


def get_residual_block(num_res, in_channels, out_channels, down_first=False):
    blk=[]
    for i in range(num_res):
        blk.append(Residual(in_channels=in_channels, out_channels=out_channels, use_1x1_conv=(i==0 and down_first), stride=1+int(i==0 and down_first)))
        in_channels=out_channels
    return blk


def resnet(in_channels=3):
    b1=nn.Sequential(
        nn.Conv2d(in_channels,16,3,1,1),
        nn.BatchNorm2d(16),
        nn.ReLU()
        # nn.MaxPool2d(3,1,1)
    )
    b2=nn.Sequential(*get_residual_block(14,16,16))
    b3=nn.Sequential(*get_residual_block(14,16,32,True))
    b4 = nn.Sequential(*get_residual_block(14,32, 64, True))
    # b5 = nn.Sequential(*get_residual_block(15,32, 64, True))
    net=nn.Sequential(b1,b2,b3,b4,
                      nn.AdaptiveAvgPool2d((1,1)),
                      nn.Flatten(),
                      nn.Linear(64,10))

    return net