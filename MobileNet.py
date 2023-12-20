from torch import nn

class Mobile(nn.Module):
    def __init__(self,num_classes=10,a=1,b=1):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.depth_spara=nn.Sequential(
            self.mobile_block(32,64,3,1),
            self.mobile_block(64, 128, 3,2),
            self.mobile_block(128, 128, 3, 1),
            self.mobile_block(128, 256, 3, 2),
            self.mobile_block(256, 256, 3, 1),
            self.mobile_block(256, 512, 3, 2),
            self.mobile_block(512, 512, 3, 1),
            self.mobile_block(512, 512, 3, 1),
            self.mobile_block(512, 512, 3, 1),
            self.mobile_block(512, 512, 3, 1),
            self.mobile_block(512, 512, 3, 1),
            self.mobile_block(512, 1024, 3, 2),
            self.mobile_block(1024, 1024, 3, 1),
            nn.AvgPool2d(1,1)
        )

        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024,10)
        )



    def mobile_block(self,in_channels,out_channels,kernel_size=3,stride=1):
        result=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=kernel_size,stride=stride,groups=in_channels,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
        return result


    def forward(self,x):
        x=self.first_conv(x)
        x=self.depth_spara(x)
        x=self.fc(x)

        return x

