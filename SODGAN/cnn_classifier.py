from torch import nn
import torch.nn.functional as F
from torchstat import stat

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Upsample):
            pass

class CNN_Classifier(nn.Module):
    def __init__(self, in_ch, n_class, size='M'):
        super().__init__()

        assert size in ['T','S', 'M', 'L']

        dilations = {
            'T': [1, 1, 1],
            'S': [1, 2, 1, 2, 1],
            'M': [1, 2, 4, 1, 2, 4, 1],
            'L': [1, 2, 4, 8, 1, 2, 4, 8, 1],
        }[size]

        channels = {
            'T': [128, 32],
            'S': [128, 64, 64, 32],
            'M': [128, 64, 64, 64, 64, 32],
            'L': [128, 64, 64, 64, 64, 64, 64, 32],
        }[size]
        channels = [in_ch] + channels + [n_class]

        layers = []
        for d, c_in, c_out in zip(dilations, channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=d, dilation=d))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers[:-1])

        self.initialize()

    def forward(self, x):
        return self.layers(x)

    def initialize(self):
        weight_init(self)



if __name__ == '__main__':
    import torch
    input = torch.rand(4096, 7040)
    classifier = CNN_Classifier(7040, 2)

    stat(classifier, (7040, 64, 64))

    print(classifier)

    output = classifier(input.unsqueeze(0).reshape(1, 64 , 64, 7040).transpose(1, 3)).squeeze().reshape(2, -1).transpose(0, 1)
    print(output.shape)