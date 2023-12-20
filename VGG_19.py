from torch import nn


class VGG(nn.Module):
    def __init__(self, conv_arch,in_channel=3,num_classes=10, init_weights=False,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv_arch=conv_arch
        self.vgg=nn.Sequential()
        for (nums,out_channel) in conv_arch:
            self.vgg.append(self.get_vgg_block(nums, out_channel, in_channel))
            in_channel=out_channel
        self.vgg.append(
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=512, out_features=4096),
                nn.ReLU(True),
                nn.Dropout(p=0.4),
                nn.ReLU(True),
                nn.Dropout(p=0.4),
                nn.Linear(4096, out_features=num_classes)
            )
        )
        if init_weights:
            self._initialize_weights()

    def get_vgg_block(self,nums,out_channel,in_channel):
        sequence=nn.Sequential()

        for i in range(nums):
            sequence.append(nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1))
            in_channel=out_channel
        sequence.append(nn.ReLU(True))
        sequence.append(nn.MaxPool2d(2, 2))

        return sequence
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x=self.vgg(x)
        return x

if __name__ == '__main__':
    conv_arch = ((1, 64), (1, 128), (2, 256))
    vgg=VGG(conv_arch,3,10)


