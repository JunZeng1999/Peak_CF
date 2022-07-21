import torch.nn as nn
import torch


class Classifier(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):
        super(Classifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, stride=2, padding=2),  # input[3, 354, 472]  output[128, 177, 236][c, h, w]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 88, 118]
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # output[256, 44, 59]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[256, 22, 29]
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2),  # output[256, 11, 14]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # output[512, 11, 14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[512, 5, 7]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 5 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
